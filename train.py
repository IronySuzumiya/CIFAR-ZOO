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

parser = argparse.ArgumentParser(description='PyTorch CIFAR Dataset Training')
parser.add_argument('--work-path', required=True, type=str)
parser.add_argument('--resume', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--stat', action='store_true',
                    help='show the statistic result of trained model')

args = parser.parse_args()
logger = Logger(log_file_name=args.work_path + '/log.txt',
                log_level=logging.DEBUG, logger_name="CIFAR").get_log()


def train(train_loader, net, criterion, optimizer, epoch, device):
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


def save_checkpoint_(net, acc, epoch, optimizer, ckpt_file_name):
    global best_prec

    state = {
        'state_dict': net.state_dict(),
        'best_prec': best_prec,
        'last_epoch': epoch,
        'optimizer': optimizer.state_dict(),
    }
    is_best = acc > best_prec
    save_checkpoint(state, is_best, ckpt_file_name)
    if is_best:
        best_prec = acc

def update_dict(dict, n):
    if n in dict:
        dict[n] += 1
    else:
        dict[n] = 1

def show_statistic_result(model):
    global config

    n_ou_with_nonzero = {}
    n_ou_with_positive = {}
    n_ou_with_negative = {}
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            rram_proj = param.view(param.shape[0], -1).T
            for i in range((rram_proj.shape[0] - 1) // config.pruning.ou_height + 1):
                for j in range((rram_proj.shape[1] - 1) // config.pruning.ou_width + 1):
                    ou = rram_proj[i * config.pruning.ou_height : (i + 1) * config.pruning.ou_height, j * config.pruning.ou_width : (j + 1) * config.pruning.ou_width]
                    update_dict(n_ou_with_nonzero, ou.nonzero().shape[0])
                    update_dict(n_ou_with_positive, (ou > 0).nonzero().shape[0])
                    update_dict(n_ou_with_negative, (ou < 0).nonzero().shape[0])
    logger.info("   == n_ou_with_nonzero: {}".format(n_ou_with_nonzero))
    logger.info("   == n_ou_with_positive: {}".format(n_ou_with_positive))
    logger.info("   == n_ou_with_negative: {}".format(n_ou_with_negative))

def main():
    global args, config, last_epoch, best_prec, writer
    writer = SummaryWriter(log_dir=args.work_path + '/event')

    # read config from yaml file
    with open(args.work_path + '/config.yaml') as f:
        config = yaml.load(f)
    # convert to dict
    config = EasyDict(config)
    logger.info(config)

    # define netowrk
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
        assert config.pruning.method == 'ADMM'
        assert config.epochs == config.pruning.pre_epochs + config.pruning.epochs + config.pruning.re_epochs
        admm_criterion = ADMMLoss(net, device, config.pruning.rho,
            config.pruning.ou_height, config.pruning.ou_width, config.pruning.percent)

    optimizer = torch.optim.SGD(
        net.parameters(),
        config.lr_scheduler.base_lr,
        momentum=config.optimize.momentum,
        weight_decay=config.optimize.weight_decay,
        nesterov=config.optimize.nesterov)

    # resume from a checkpoint
    last_epoch = -1
    best_prec = 0
    if args.work_path:
        if 'pruning' in config:
            ckpt_name = "{}_{}x{}".format(config.ckpt_name, config.pruning.ou_height, config.pruning.ou_width)
        else:
            ckpt_name = config.ckpt_name
        ckpt_file_name = args.work_path + '/' + ckpt_name + '.pth.tar'
        if args.resume:
            best_prec, last_epoch = load_checkpoint(
                ckpt_file_name, net, optimizer=optimizer)

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
                    save_checkpoint_(net, test_acc * 100., epoch, optimizer, ckpt_file_name)
            logger.info(
                "======== Pre-Training Finished.   best_test_acc: {:.3f}% ========\n".format(best_prec))

        if begin_epoch < config.pruning.pre_epochs + config.pruning.epochs:
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
                    save_checkpoint_(net, test_acc * 100., epoch, optimizer, ckpt_file_name)
            logger.info(
                "======== Training Finished.   best_test_acc: {:.3f}% ========\n".format(best_prec))

            logger.info("            =======  Applying ADMM Pruning  =======")
            admm_criterion.apply_pruning()
            prune_param, total_param = 0, 0
            for name, param in net.named_parameters():
                if name.split('.')[-1] == "weight":
                    logger.info("   ==  [at weight {}]  ==".format(name))
                    logger.info("   ==  percentage of pruned: {:.4f}%  ==".format(100 * (abs(param) == 0).sum().item() / param.numel()))
                    logger.info("   ==  nonzero parameters after pruning: {} / {}  ==".format((param != 0).sum().item(), param.numel()))
                total_param += param.numel()
                prune_param += (param != 0).sum().item()
            logger.info("   ==  Total nonzero parameters after pruning: {} / {} ({:.4f}%)  ==\n".
                format(prune_param, total_param, 100 * (total_param - prune_param) / total_param))

        if begin_epoch < config.epochs:
            retrain_begin_epoch = max(begin_epoch, config.pruning.pre_epochs + config.pruning.epochs)
            logger.info("            =======  Re-Training  =======\n")
            for epoch in range(retrain_begin_epoch, config.epochs):
                lr = adjust_learning_rate(optimizer, epoch, config)
                writer.add_scalar('learning_rate', lr, epoch)
                train(train_loader, net, criterion, optimizer, epoch, device)
                if epoch == config.pruning.pre_epochs + config.pruning.epochs or \
                        (epoch + 1 - config.pruning.pre_epochs - config.pruning.epochs) % config.eval_freq == 0 or \
                        epoch == config.epochs - 1:
                    _, test_acc = test(test_loader, net, criterion, optimizer, epoch, device)
                    save_checkpoint_(net, test_acc * 100., epoch, optimizer, ckpt_file_name)
            logger.info(
                "======== Re-Training Finished.   best_test_acc: {:.3f}% ========\n".format(best_prec))

        if args.stat:
            logger.info("            =======  Showing Statistic Result  =======\n")
            show_statistic_result(net)
        
    else:
        logger.info("            =======  Training  =======\n")
        for epoch in range(last_epoch + 1, config.epochs):
            lr = adjust_learning_rate(optimizer, epoch, config)
            writer.add_scalar('learning_rate', lr, epoch)
            train(train_loader, net, criterion, optimizer, epoch, device)
            if epoch == 0 or (epoch + 1) % config.eval_freq == 0 or epoch == config.epochs - 1:
                _, test_acc = test(test_loader, net, criterion, optimizer, epoch, device)
                save_checkpoint_(net, test_acc * 100., epoch, optimizer, ckpt_file_name)
        logger.info(
        "======== Training Finished.   best_test_acc: {:.3f}% ========".format(best_prec))
    writer.close()


if __name__ == "__main__":
    main()
