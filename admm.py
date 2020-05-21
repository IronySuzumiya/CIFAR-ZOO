import torch
import torch.nn as nn
import torch.nn.functional as F
import admm_cuda_lib
from collections import Counter

class ADMMLoss(nn.Module):
    def __init__(self, model, device, rho, percent, mode):
        super().__init__()
        self.model = model
        self.device = device
        self.rho = rho
        self.percent = percent
        self.Z = ()
        self.U = ()
        self.dict_mask = {}
        self.fkw = []
        self.mode = mode
        self.patterns = []
        self.update_Z = self.update_Z_unstructured if mode == 'unstructured' \
            else self.update_Z_grid_based if mode == 'grid-based' \
            else self.update_Z_pattern_based if mode == 'pattern-based' \
            else self.update_Z_unimplemented
        self.apply_pruning = self.apply_pruning_unstructured if mode == 'unstructured' \
            else self.apply_pruning_grid_based if mode == 'grid-based' \
            else self.apply_pruning_pattern_based if mode == 'pattern-based' \
            else self.apply_pruning_unimplemented
        for name, param in model.named_parameters():
            if name.split('.')[-1] == "weight" and len(param.shape) == 4:
                assert(param.shape[2] == param.shape[3])
                self.Z += (param.detach().clone().to(device),)
                self.U += (torch.zeros_like(param).to(device),)
        
    def forward(self, input, target):
        idx = 0
        loss = F.cross_entropy(input, target)
        for name, param in self.model.named_parameters():
            if name.split('.')[-1] == "weight" and len(param.shape) == 4:
                u = self.U[idx]
                z = self.Z[idx]
                loss += self.rho / 2 * (param - z + u).norm()
                idx += 1
        return loss

    def get_state(self):
        return self.Z, self.U, self.dict_mask, self.patterns, self.fkw

    def load_state(self, state):
        self.Z, self.U, self.dict_mask, self.patterns, self.fkw = state

    def get_mask(self):
        return self.dict_mask

    def get_fkw(self):
        return self.fkw

    def is_natural_patterns_extracted(self):
        return self.patterns

    def extract_natural_patterns(self, size_pattern, percent, num_patterns):
        pattern_list = []
        idx = 0
        for name, param in self.model.named_parameters():
            if name.split('.')[-1] == "weight" and len(param.shape) == 4:
                kernel_abs = param.detach().view(-1, param.shape[2] ** 2).abs()
                values, indices = kernel_abs.topk(size_pattern)
                norm_values = values.norm(dim=1)
                _, topk_indices = norm_values.topk(round((1 - percent[idx]) * norm_values.numel()))
                final_indices = indices[topk_indices]
                # centre value must be in the pattern
                for i in range(final_indices.shape[0]):
                    if param.shape[2] ** 2 // 2 not in final_indices[i]:
                        final_indices[i][-1] = param.shape[2] ** 2 // 2
                final_indices, _ = final_indices.sort()
                pattern_list.extend(final_indices.tolist())
                idx += 1
        pattern_dict = Counter(list(map(lambda x: tuple(x), pattern_list)))
        sorted_pattern_dict = sorted(pattern_dict.items(), key=lambda x: x[1], reverse=True)
        patterns, count = zip(*sorted_pattern_dict[:num_patterns])
        self.patterns = patterns

    def set_grid_size(self, grid_height, grid_width):
        self.grid_height = grid_height
        self.grid_width = grid_width

    def update_ADMM(self):
        self.update_X()
        self.update_Z()
        self.update_U()

    def update_X(self):
        self.X = ()
        for name, param in self.model.named_parameters():
            if name.split('.')[-1] == "weight" and len(param.shape) == 4:
                self.X += (param.detach(),)

    def update_Z_unstructured(self):
        self.Z = ()
        idx = 0
        for x, u in zip(self.X, self.U):
            z = x + u
            pcen, _ = abs(z.view(-1)).kthvalue(round(self.percent[idx] * z.numel()))
            under_threshold = abs(z) < pcen
            z.data[under_threshold] = 0
            self.Z += (z,)
            idx += 1

    def update_Z_grid_based(self):
        self.Z = ()
        idx = 0
        for x, u in zip(self.X, self.U):
            z = x + u
            rram = z.view(z.shape[0], -1)
            tmp = torch.zeros(((rram.shape[0] - 1) // self.grid_width + 1, (rram.shape[1] - 1) // self.grid_height + 1)).to(self.device)
            admm_cuda_lib.struct_norm(rram, tmp, self.grid_width, self.grid_height)
            pcen, _ = tmp.view(-1).kthvalue(round(self.percent[idx] * tmp.numel()))
            upon_threshold = tmp >= pcen
            res1 = rram.shape[0] % self.grid_width
            res2 = rram.shape[1] % self.grid_height
            for i in range(self.grid_width):
                for j in range(self.grid_height):
                    if i < res1 or res1 == 0:
                        rram.data[i::self.grid_width, j::self.grid_height] *= upon_threshold if j < res2 or res2 == 0 else upon_threshold[:, :-1]
                    else:
                        rram.data[i::self.grid_width, j::self.grid_height] *= upon_threshold[:-1, :] if j < res2 or res2 == 0 else upon_threshold[:-1, :-1]
            self.Z += (z,)
            idx += 1

    def update_Z_pattern_based(self):
        self.Z = ()
        idx = 0
        num_patterns = len(self.patterns)
        for x, u in zip(self.X, self.U):
            z = x + u
            z_flatten = z.view(-1, z.shape[2] ** 2)
            norm_values = z_flatten.norm(dim=1)
            _, topk_indices = norm_values.topk(round((1 - self.percent[idx]) * norm_values.numel()))
            z_flatten[list(set(range(z_flatten.shape[0])) - set(topk_indices.tolist())), :] = 0
            z_flatten_topk = z_flatten[topk_indices, :]
            pattern_compat = torch.zeros(topk_indices.numel(), num_patterns).type_as(z).to(self.device)
            for i in range(num_patterns):
                pattern_compat[:, i] = z_flatten_topk[:, self.patterns[i]].norm(dim=1)
            best_patterns = pattern_compat.argmax(1)
            for i in range(best_patterns.numel()):
                z_flatten_topk[i, list(set(range(z_flatten.shape[1])) - set(self.patterns[best_patterns[i]]))] = 0
            self.Z += (z,)
            idx += 1

    def update_Z_unimplemented(self):
        RuntimeError("Pruning Mode '{}' Unimplemented!".format(self.mode))

    def update_U(self):
        new_U = ()
        for u, x, z in zip(self.U, self.X, self.Z):
            new_u = u + x - z
            new_U += (new_u,)
        self.U = new_U

    def calc_convergence(self):
        idx = 0
        res_list = []
        for name, param in self.model.named_parameters():
            if name.split('.')[-1] == "weight" and len(param.shape) == 4:
                x, z = self.X[idx], self.Z[idx]
                res_list.append((name, (x-z).norm().item() / x.norm().item()))
                idx += 1
        return res_list

    def is_pruning_applied(self):
        return self.fkw and self.dict_mask

    def apply_pruning_unstructured(self):
        self.dict_mask = {}
        self.fkw = []
        idx = 0

        for name, param in self.model.named_parameters():
            if name.split('.')[-1] == "weight" and len(param.shape) == 4:
                pcen, _ = abs(param.view(-1)).kthvalue(round(self.percent[idx] * param.numel()))
                mask = (abs(param) >= pcen).to(self.device)
                param.data.mul_(mask)
                self.dict_mask[name] = mask
                idx += 1
                self.fkw.append([])
                rram_mask = mask.view(msak.shape[0], -1)
                for i in range(rram_mask.shape[0]):
                    self.fkw[-1].append(rram_mask[i, :].nonzero().view(-1).tolist())

    def apply_pruning_grid_based(self):
        self.dict_mask = {}
        self.fkw = []
        idx = 0

        for name, param in self.model.named_parameters():
            if name.split('.')[-1] == "weight" and len(param.shape) == 4:
                mask = torch.zeros_like(param, dtype=torch.bool).to(self.device)
                rram = param.view(param.shape[0], -1)
                rram_mask = mask.view(mask.shape[0], -1)
                norm = torch.zeros(((rram.shape[0] - 1) // self.grid_width + 1, (rram.shape[1] - 1) // self.grid_height + 1)).to(self.device)
                admm_cuda_lib.struct_norm(rram, norm, self.grid_width, self.grid_height)
                pcen, _ = norm.view(-1).kthvalue(round(self.percent[idx] * norm.numel()))
                upon_threshold = norm >= pcen
                res1 = rram.shape[0] % self.grid_width
                res2 = rram.shape[1] % self.grid_height
                for i in range(self.grid_width):
                    for j in range(self.grid_height):
                        if i < res1 or res1 == 0:
                            rram_mask.data[i::self.grid_width, j::self.grid_height] = upon_threshold if j < res2 or res2 == 0 else upon_threshold[:, :-1]
                        else:
                            rram_mask.data[i::self.grid_width, j::self.grid_height] = upon_threshold[:-1, :] if j < res2 or res2 == 0 else upon_threshold[:-1, :-1]
                param.data.mul_(mask)
                self.dict_mask[name] = mask
                idx += 1
                self.fkw.append([])
                for i in range(upon_threshold.shape[0]):
                    self.fkw[-1].append(upon_threshold[i, :].nonzero().view(-1).tolist())

    def apply_pruning_pattern_based(self):
        self.dict_mask = {}
        self.fkw = []
        idx = 0
        num_patterns = len(self.patterns)

        for name, param in self.model.named_parameters():
            if name.split('.')[-1] == "weight" and len(param.shape) == 4:
                weight = param.detach()
                mask = torch.ones_like(weight).to(self.device)
                weight_flatten = weight.view(-1, weight.shape[2] ** 2)
                mask_flatten = mask.view(-1, mask.shape[2] ** 2)
                norm_values = weight_flatten.norm(dim=1)
                _, topk_indices = norm_values.topk(round((1 - self.percent[idx]) * norm_values.numel()))
                topk_indices, _ = topk_indices.sort()
                mask_flatten[list(set(range(mask_flatten.shape[0])) - set(topk_indices.tolist())), :] = 0
                weight_flatten_topk = weight_flatten[topk_indices, :]
                mask_flatten_topk = mask_flatten[topk_indices, :]
                pattern_compat = torch.zeros(topk_indices.numel(), num_patterns).type_as(weight).to(self.device)
                for i in range(num_patterns):
                    pattern_compat[:, i] = weight_flatten_topk[:, self.patterns[i]].norm(dim=1)
                best_patterns = pattern_compat.argmax(1)
                for i in range(best_patterns.numel()):
                    mask_flatten_topk[i, list(set(range(mask_flatten.shape[1])) - set(self.patterns[best_patterns[i]]))] = 0
                param.data.mul_(mask)
                self.dict_mask[name] = mask
                idx += 1
                self.fkw.append([])
                fkw = list(zip(topk_indices.tolist(), best_patterns.tolist()))
                fkw = list(map(lambda x: (x[0] // param.shape[1], x[0] % param.shape[1], x[1]), fkw))
                for i in range(param.shape[0]):
                    tmp = list(filter(lambda x: x[0] == i, fkw))
                    self.fkw[-1].append(list(map(lambda x: (x[1], x[2]), tmp)))\
                        
    def apply_pruning_unimplemented(self):
        RuntimeError("Pruning Mode '{}' Unimplemented!".format(self.mode))
