# -*-coding:utf-8-*-
import argparse
import logging
import yaml
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter

from easydict import EasyDict
from models import *

from utils import Logger, count_parameters, data_augmentation, \
    load_checkpoint, get_data_loader, mixup_data, mixup_criterion, \
    save_checkpoint, adjust_learning_rate, get_current_lr

from admm import ADMMLoss

import multiprocessing as mp
import traceback

from optimizer import PruneSGD

import numpy as np
import random

from lcs import LCS

from copy import deepcopy

parser = argparse.ArgumentParser(description='PyTorch CIFAR Dataset Training')
parser.add_argument('--work-path', required=True, type=str)
parser.add_argument('--restart', action='store_true',
                    help='start from scratch')
parser.add_argument('--display-fkwmd', action='store_true',
                    help='display matching degree between each FKW after reordering')
parser.add_argument('--calc-save-bcm', action='store_true',
                    help='calculate and save best channel matches')
parser.add_argument('--display-bcm', action='store_true',
                    help='display best channel matches')
parser.add_argument('--bitsW', type=int, default=8, metavar='b',
                    help='weight bits (default: 8)')
#parser.add_argument('--display_cprs', action='store_true',
#                    help='display the compressed trained model')

args = parser.parse_args()
logger = Logger(log_file_name=args.work_path + '/log.txt',
                log_level=logging.DEBUG, logger_name="CIFAR").get_log()


def train(train_loader, net, criterion, optimizer, epoch, device, mask=None):
    global writer

    start = time.time()
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    logger.info(" === Epoch: [{}/{}] === ".format(epoch + 1, config.epochs))

    for batch_index, (inputs, targets) in enumerate(train_loader):
        # move tensor to GPU
        inputs, targets = inputs.to(device), targets.to(device)
        if config.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, config.mixup_alpha, device)

            outputs = net(inputs)
            loss = mixup_criterion(
                criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        # zero the gradient buffers
        optimizer.zero_grad()
        # backward
        loss.backward()
        # update weight
        if mask:
            optimizer.prune_step(mask)
        else:
            optimizer.step()

        # count the loss and acc
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if config.mixup:
            correct += (lam * predicted.eq(targets_a).sum().item()
                        + (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            correct += predicted.eq(targets).sum().item()

        if (batch_index + 1) % 100 == 0:
            logger.info("   == step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}".format(
                batch_index + 1, len(train_loader),
                train_loss / (batch_index + 1), 100.0 * correct / total, get_current_lr(optimizer)))

    logger.info("   == step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}".format(
        batch_index + 1, len(train_loader),
        train_loss / (batch_index + 1), 100.0 * correct / total, get_current_lr(optimizer)))

    end = time.time()
    logger.info("   == cost time: {:.4f}s".format(end - start))
    train_loss = train_loss / (batch_index + 1)
    train_acc = correct / total

    writer.add_scalar('train_loss', train_loss, global_step=epoch)
    writer.add_scalar('train_acc', train_acc, global_step=epoch)

    return train_loss, train_acc


def test_(test_loader, net, criterion, device):
    global writer

    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    logger.info(" === Validate ===")

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    logger.info("   == test loss: {:.3f} | test acc: {:6.3f}%".format(
        test_loss / (batch_index + 1), 100.0 * correct / total))


def test(test_loader, net, criterion, optimizer, epoch, device):
    global writer

    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    logger.info(" === Validate ===".format(epoch + 1, config.epochs))

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    logger.info("   == test loss: {:.3f} | test acc: {:6.3f}%".format(
        test_loss / (batch_index + 1), 100.0 * correct / total))
    test_loss = test_loss / (batch_index + 1)
    test_acc = correct / total
    writer.add_scalar('test_loss', test_loss, global_step=epoch)
    writer.add_scalar('test_acc', test_acc, global_step=epoch)

    return test_loss, test_acc


def save_checkpoint_(net, acc, epoch, optimizer, admm_state, ckpt_name):
    global args, best_prec

    state = {
        'state_dict': net.state_dict(),
        'best_prec': best_prec,
        'last_epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'admm_state_dict': admm_state,
    }
    is_best = acc > best_prec
    save_checkpoint(state, is_best, args.work_path + '/' + ckpt_name)
    if is_best:
        best_prec = acc

def load_checkpoint_(ckpt_name, net, optimizer=None, admm_criterion=None):
    global args

    load_checkpoint(args.work_path + '/' + ckpt_name + '.pth.tar', net, optimizer, admm_criterion)

def quantize_weights(weight):
    global args

    h_lvl = 2 ** (args.bitsW - 1) - 1
    delta_w = weight.abs().max() / h_lvl
    return torch.round(weight / delta_w)

def display_fkwmd(model, admm_criterion, show_seq=False):
    idx = 0
    fkw = admm_criterion.get_fkw()
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4:
            logger.info(name)

            len_subseqs = torch.zeros((len(fkw[idx]), len(fkw[idx])), dtype=torch.int)
            longest_subseqs = []

            longest_subseq = []
            longest_subseq_idx = 1
            for j in range(1, len(fkw[idx])):
                subseq, _, _ = LCS(fkw[idx][0], fkw[idx][j])
                len_subseqs[0, j] = len(subseq)
                if len(subseq) > len(longest_subseq):
                    longest_subseq = subseq
                    longest_subseq_idx = j
            longest_subseqs.append((longest_subseq_idx, longest_subseq))

            for i in range(1, len(fkw[idx]) - 1):
                longest_subseq = []
                longest_subseq_idx = i + 1
                for j in range(i+1, len(fkw[idx])):
                    subseq, _, _ = LCS(fkw[idx][i], fkw[idx][j])
                    len_subseqs[i, j] = len(subseq)
                    if len(subseq) > len(longest_subseq):
                        longest_subseq = subseq
                        longest_subseq_idx = j
                pre_max_len, pre_max_len_index = len_subseqs[:i, i].max(dim=0)
                if len(longest_subseq) < pre_max_len.item():
                    longest_subseq_idx = pre_max_len_index.item()
                    subseq, _, _ = LCS(fkw[idx][i], fkw[idx][longest_subseq_idx])
                    longest_subseq = subseq
                longest_subseqs.append((longest_subseq_idx, longest_subseq))
            
            logger.info("len subseqs matrix:\n{}".format(len_subseqs))
            for i in range(len(longest_subseqs)):
                seq_idx = longest_subseqs[i][0]
                seq = longest_subseqs[i][1]
                len_subseq = len_subseqs[i, seq_idx] if i < seq_idx else len_subseqs[seq_idx, i]
                len_i = len(fkw[idx][i])
                len_seq_idx = len(fkw[idx][seq_idx])
                pcen0 = len_subseq * 100.0 / len_i if len_i > 0 else 100.0
                pcen1 = len_subseq * 100.0 / len_seq_idx if len_seq_idx > 0 else 100.0
                logger.info("fkw[{}] & fkw[{}]: len = {}, pcen = ({:.1f}%, {:.1f}%)"
                    .format(i, seq_idx, len_subseq, pcen0, pcen1))
                if show_seq:
                    logger.info("  seq: {}".format(seq))
            idx += 1

def calc_and_save_best_channel_matches_(fkw, name, filename):
    logger.info("=====  {} running...  =====".format(name))

    matches = []

    num_total_units = 0
    for i in range(len(fkw)):
        num_total_units += len(fkw[i])

    for i in range(0, len(fkw) - 1):
        while len(fkw[i]):
            longest_subseq = []
            longest_subseq_idx = i + 1
            for j in range(i + 1, len(fkw)):
                subseq, subseqid1, subseqid2 = LCS(fkw[i], fkw[j])
                if len(subseq) > len(longest_subseq):
                    longest_subseq = subseq
                    longest_subseq_idx = j
            if len(longest_subseq) == 0:
                break
            matches.append((i, longest_subseq_idx, len(longest_subseq), longest_subseq))
            for j in reversed(subseqid1):
                del fkw[i]
            for j in reversed(subseqid2):
                del fkw[longest_subseq_idx]
    
    num_left_units = 0
    for i in range(len(fkw)):
        num_left_units += len(fkw[i])

    left_4_dead_percent = num_left_units * 100.0 / num_total_units if num_total_units else 0.0

    torch.save((matches, left_4_dead_percent), filename)

    logger.info("=====  {} done.  =====".format(name))

def calc_and_save_best_channel_matches(model, admm_criterion, ckpt_name):
    idx = 0
    fkw = deepcopy(admm_criterion.get_fkw())
    pool = mp.Pool(processes=16)
    process_results = []
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4:
            file_name = args.work_path + '/' + ckpt_name + '_' + name + '.bcm'
            process_results.append(
                pool.apply_async(calc_and_save_best_channel_matches_, (fkw[idx], name, file_name))
            )
            idx += 1
    
    pool.close()
    pool.join()

    idx = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4:
            try:
                process_results[idx].get()
            except:
                with open(args.work_path + '/error.txt', 'w') as error_file:
                    traceback.print_exc(file=error_file)
            idx += 1

    logger.info("=====  All done.  =====")

def display_best_channel_matches(model, ckpt_name, display_subseq=False):
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4:
            logger.info("=====  {}  =====".format(name))

            file_name = args.work_path + '/' + ckpt_name + '_' + name + '.bcm'
            matches, left_4_dead_percent = torch.load(file_name)

            logger.info("Matches List:")

            if not display_subseq:
                matches_withnot_subseq = list(map(lambda x: x[:3], matches))
                logger.info(matches_withnot_subseq)
            else:
                logger.info(matches)

            logger.info("Left Percent: {:.1f}%".format(left_4_dead_percent))


def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   torch.cuda.manual_seed(seed)
   np.random.seed(seed)
   random.seed(seed)
   cudnn.deterministic = True

def main():
    global args, config, last_epoch, best_prec, writer

    writer = SummaryWriter(log_dir=args.work_path + '/event')

    # read config from yaml file
    with open(args.work_path + '/config.yaml') as f:
        config = yaml.load(f)
    # convert to dict
    config = EasyDict(config)
    logger.info(config)

    setup_seed(2020)

    # define network
    net = get_model(config)
    logger.info(net)
    logger.info(" == total parameters: " + str(count_parameters(net)))

    # CPU or GPU
    device = 'cuda' if config.use_gpu else 'cpu'
    # data parallel for multiple-GPU
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    net.to(device)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if 'pruning' in config:
        assert(config.epochs == config.pruning.pre_epochs + config.pruning.epochs + config.pruning.re_epochs,
            "ADMM-3-stage epochs must equal to total epochs")
        admm_criterion = ADMMLoss(net, device, config.pruning.rho, config.pruning.percent, config.pruning.mode)
        optimizer = PruneSGD(
            net.named_parameters(),
            config.lr_scheduler.base_lr,
            momentum=config.optimize.momentum,
            weight_decay=config.optimize.weight_decay,
            nesterov=config.optimize.nesterov)
    else:
        admm_criterion = None
        optimizer = torch.optim.SGD(
            net.parameters(),
            config.lr_scheduler.base_lr,
            momentum=config.optimize.momentum,
            weight_decay=config.optimize.weight_decay,
            nesterov=config.optimize.nesterov,
            signed=config.pruning.mode == 'grid-based-sign-separate' and (config.pruning.grid_height != 1 or config.pruning.grid_width != 1))

    # resume from a checkpoint
    if 'pruning' in config:
        if config.pruning.mode == 'unstructured':
            ckpt_name = "{}_unstruct".format(config.ckpt_name)
        elif config.pruning.mode == 'grid-based':
            ckpt_name = "{}_grid_{}x{}".format(config.ckpt_name, config.pruning.grid_height, config.pruning.grid_width)
        elif config.pruning.mode == 'pattern-based':
            ckpt_name = "{}_pattern_{}x{}".format(config.ckpt_name, config.pruning.size_pattern, config.pruning.num_patterns)
        elif config.pruning.mode == 'grid-based-sign-separate':
            ckpt_name = "{}_signed_grid_{}x{}".format(config.ckpt_name, config.pruning.size_pattern, config.pruning.num_patterns)
    else:
        ckpt_name = config.ckpt_name
    ckpt_file_name = args.work_path + '/' + ckpt_name + '.pth.tar'
    if not args.restart:
        best_prec, last_epoch = load_checkpoint(
            ckpt_file_name, net, optimizer=optimizer, admm_criterion=admm_criterion)
    else:
        best_prec, last_epoch = 0, -1

    # load training data, do data augmentation and get data loader
    transform_train = transforms.Compose(
        data_augmentation(config))

    transform_test = transforms.Compose(
        data_augmentation(config, is_train=False))

    train_loader, test_loader = get_data_loader(
        transform_train, transform_test, config)

    if 'pruning' in config:
        begin_epoch = last_epoch + 1
        if begin_epoch < config.pruning.pre_epochs:
            logger.info("            =======  Pre-Training  =======\n")
            for epoch in range(begin_epoch, config.pruning.pre_epochs):
                lr = adjust_learning_rate(optimizer, epoch, config)
                writer.add_scalar('learning_rate', lr, epoch)
                train(train_loader, net, criterion, optimizer, epoch, device)
                if epoch == 0 or (epoch + 1) % config.eval_freq == 0 or epoch == config.pruning.pre_epochs - 1:
                    _, test_acc = test(test_loader, net, criterion, optimizer, epoch, device)
                    save_checkpoint_(net, test_acc * 100., epoch, optimizer, admm_criterion.get_state(), ckpt_name)
            logger.info(
                "======== Pre-Training Finished.   best_test_acc: {:.3f}% ========\n".format(best_prec))

        if begin_epoch < config.pruning.pre_epochs + config.pruning.epochs:
            if config.pruning.mode == 'pattern-based' and not admm_criterion.is_natural_patterns_extracted():
                # only consider 3x3 weight kernel
                logger.info("            =======  Extracting Natural Patterns  =======\n")
                admm_criterion.extract_natural_patterns(
                    config.pruning.size_pattern, config.pruning.percent, config.pruning.num_patterns)
                logger.info("======== Extracting Natural Patterns Finished. ========\n")
            elif config.pruning.mode == 'grid-based':
                admm_criterion.set_grid_size(config.pruning.grid_height, config.pruning.grid_width)

            admm_begin_epoch = max(begin_epoch, config.pruning.pre_epochs)
            logger.info("            =======  Training with ADMM Pruning  =======\n")
            for epoch in range(admm_begin_epoch, config.pruning.pre_epochs + config.pruning.epochs):
                lr = adjust_learning_rate(optimizer, epoch, config)
                writer.add_scalar('learning_rate', lr, epoch)
                train(train_loader, net, admm_criterion, optimizer, epoch, device)
                logger.info("   ==  Updating ADMM State  ==")
                admm_criterion.update_ADMM()
                logger.info("   ==  Normalized norm of (weight - projection)  ==")
                res_list = admm_criterion.calc_convergence()
                for name, convrg in res_list:
                    logger.info("   ==  ({}): {:.4f}  ==".format(name, convrg))
                if epoch == admm_begin_epoch or (epoch + 1 - admm_begin_epoch) % config.eval_freq == 0 \
                        or epoch == config.pruning.pre_epochs + config.pruning.epochs - 1:
                    _, test_acc = test(test_loader, net, criterion, optimizer, epoch, device)
                    save_checkpoint_(net, test_acc * 100., epoch, optimizer, admm_criterion.get_state(), ckpt_name)
            logger.info(
                "======== Training with ADMM Pruning Finished.   best_test_acc: {:.3f}% ========\n".format(best_prec))

        if begin_epoch < config.epochs:
            if not admm_criterion.is_pruning_applied():
                logger.info("            =======  Applying ADMM Pruning  =======\n")
                admm_criterion.apply_pruning()
                prune_param, total_param = 0, 0
                for name, param in net.named_parameters():
                    if name.split('.')[-1] == "weight" and len(param.shape) == 4:
                        logger.info("   ==  [at weight {}]  ==".format(name))
                        logger.info("   ==  percentage of pruned: {:.4f}%  ==".format(100 * (abs(param) == 0).sum().item() / param.numel()))
                        logger.info("   ==  nonzero parameters after pruning: {} / {}  ==".format((param != 0).sum().item(), param.numel()))
                    total_param += param.numel()
                    prune_param += (param != 0).sum().item()
                logger.info("   ==  Total nonzero parameters after pruning: {} / {} ({:.4f}%)  ==\n".
                    format(prune_param, total_param, 100 * (total_param - prune_param) / total_param))
                logger.info("======== Applying ADMM Pruning Finished. ========\n")

            retrain_begin_epoch = max(begin_epoch, config.pruning.pre_epochs + config.pruning.epochs)
            logger.info("            =======  Re-Training  =======\n")
            for epoch in range(retrain_begin_epoch, config.epochs):
                lr = adjust_learning_rate(optimizer, epoch, config)
                writer.add_scalar('learning_rate', lr, epoch)
                train(train_loader, net, criterion, optimizer, epoch, device, admm_criterion.get_mask())
                if epoch == config.pruning.pre_epochs + config.pruning.epochs or \
                        (epoch + 1 - config.pruning.pre_epochs - config.pruning.epochs) % config.eval_freq == 0 or \
                        epoch == config.epochs - 1:
                    _, test_acc = test(test_loader, net, criterion, optimizer, epoch, device)
                    save_checkpoint_(net, test_acc * 100., epoch, optimizer, admm_criterion.get_state(), ckpt_name)
            logger.info(
                "======== Re-Training Finished.   best_test_acc: {:.3f}% ========\n".format(best_prec))

        if args.display_fkwmd:
            logger.info("            =======  Displaying Matching Degree between each FKW after Reordering  =======\n")
            display_fkwmd(net, admm_criterion)

        if args.calc_save_bcm:
            logger.info("            =======  Calculating and Saving Best Channel Matches  =======\n")
            calc_and_save_best_channel_matches(net, admm_criterion, ckpt_name)

        if args.display_bcm:
            logger.info("            =======  Displaying Best Channel Matches  =======\n")
            display_best_channel_matches(net, ckpt_name)
        
    else:
        logger.info("            =======  Training  =======\n")
        for epoch in range(last_epoch + 1, config.epochs):
            lr = adjust_learning_rate(optimizer, epoch, config)
            writer.add_scalar('learning_rate', lr, epoch)
            train(train_loader, net, criterion, optimizer, epoch, device)
            if epoch == 0 or (epoch + 1) % config.eval_freq == 0 or epoch == config.epochs - 1:
                _, test_acc = test(test_loader, net, criterion, optimizer, epoch, device)
                save_checkpoint_(net, test_acc * 100., epoch, optimizer, None, ckpt_name)
        logger.info(
            "======== Training Finished.   best_test_acc: {:.3f}% ========".format(best_prec))
    writer.close()


if __name__ == "__main__":
    main()
