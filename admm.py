import torch
import torch.nn as nn
import torch.nn.functional as F
import admm_cuda_lib

class ADMMLoss(nn.Module):
    def __init__(self, model, device, rho, ou_height, ou_width, percent):
        super().__init__()
        self.model = model
        self.device = device
        self.rho = rho
        self.ou_height = ou_height
        self.ou_width = ou_width
        self.percent = percent
        self.Z = ()
        self.U = ()
        for name, param in model.named_parameters():
            if name.split('.')[-1] == "weight":
                self.Z += (param.detach().clone().to(device),)
                self.U += (torch.zeros_like(param).to(device),)
        
    def forward(self, input, target):
        idx = 0
        loss = F.cross_entropy(input, target)
        for name, param in self.model.named_parameters():
            if name.split('.')[-1] == "weight":
                u = self.U[idx]
                z = self.Z[idx]
                loss += self.rho / 2 * (param - z + u).norm()
                idx += 1
        return loss

    def get_state(self):
        return self.Z, self.U

    def load_state(self, state):
        self.Z, self.U = state

    def update_ADMM(self):
        self.update_X()
        self.update_Z()
        self.update_U()

    def update_X(self):
        self.X = ()
        for name, param in self.model.named_parameters():
            if name.split('.')[-1] == "weight":
                self.X += (param.detach(),)

    def update_Z(self):
        self.Z = ()
        idx = 0
        for x, u in zip(self.X, self.U):
            z = x + u
            if self.ou_height > 1 or self.ou_width > 1:
                rram = z.view(z.shape[0], -1)
                tmp = torch.zeros(((rram.shape[0] - 1) // self.ou_width + 1, (rram.shape[1] - 1) // self.ou_height + 1)).to(self.device)
                admm_cuda_lib.struct_norm(rram, tmp, self.ou_width, self.ou_height)
                pcen, _ = tmp.view(-1).kthvalue(round(self.percent[idx] * tmp.numel()))
                upon_threshold = tmp >= pcen
                res1 = rram.shape[0] % self.ou_width
                res2 = rram.shape[1] % self.ou_height
                for i in range(self.ou_width):
                    for j in range(self.ou_height):
                        if i < res1 or res1 == 0:
                            rram.data[i::self.ou_width, j::self.ou_height] *= upon_threshold if j < res2 or res2 == 0 else upon_threshold[:, :-1]
                        else:
                            rram.data[i::self.ou_width, j::self.ou_height] *= upon_threshold[:-1, :] if j < res2 or res2 == 0 else upon_threshold[:-1, :-1]
            else:
                pcen, _ = torch.kthvalue(abs(z.view(-1)), round(self.percent[idx] * z.numel()))
                under_threshold = abs(z) < pcen
                z.data[under_threshold] = 0
            self.Z += (z,)
            idx += 1

    def update_U(self):
        new_U = ()
        for u, x, z in zip(self.U, self.X, self.Z):
            new_u = u + x - z
            new_U += (new_u,)
        self.U = new_U

    def calc_convergence(self):
        idx = 0
        res_list = []
        for name, _ in self.model.named_parameters():
            if name.split('.')[-1] == "weight":
                x, z = self.X[idx], self.Z[idx]
                res_list.append((name, (x-z).norm().item() / x.norm().item()))
                idx += 1
        return res_list

    def apply_pruning(self):
        dict_mask = {}
        idx = 0
        for name, param in self.model.named_parameters():
            if name.split('.')[-1] == "weight":
                weight = param.detach()
                if self.ou_height > 1 or self.ou_width > 1:
                    mask = torch.zeros_like(weight, dtype=torch.bool).to(self.device)
                    rram = weight.view(weight.shape[0], -1)
                    rram_mask = mask.view(mask.shape[0], -1)
                    tmp = torch.zeros(((rram.shape[0] - 1) // self.ou_width + 1, (rram.shape[1] - 1) // self.ou_height + 1)).to(self.device)
                    admm_cuda_lib.struct_norm(rram, tmp, self.ou_width, self.ou_height)
                    pcen, _ = tmp.view(-1).kthvalue(round(self.percent[idx] * tmp.numel()))
                    upon_threshold = tmp >= pcen
                    res1 = rram.shape[0] % self.ou_width
                    res2 = rram.shape[1] % self.ou_height
                    for i in range(self.ou_width):
                        for j in range(self.ou_height):
                            if i < res1 or res1 == 0:
                                rram_mask.data[i::self.ou_width, j::self.ou_height] = upon_threshold if j < res2 or res2 == 0 else upon_threshold[:, :-1]
                            else:
                                rram_mask.data[i::self.ou_width, j::self.ou_height] = upon_threshold[:-1, :] if j < res2 or res2 == 0 else upon_threshold[:-1, :-1]
                else:
                    pcen, _ = torch.kthvalue(abs(weight.view(-1)), round(self.percent[idx] * weight.numel()))
                    mask = (abs(weight) >= pcen).to(self.device)
                param.data.mul_(mask)
                dict_mask[name] = mask
                idx += 1
        return dict_mask
