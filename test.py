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

import os

parser = argparse.ArgumentParser(description='PyTorch CIFAR Dataset Testing')
parser.add_argument('--work-path', required=True, type=str)

args = parser.parse_args()
logger = Logger(log_file_name=args.work_path + '/log.txt',
                log_level=logging.DEBUG, logger_name="CIFAR").get_log()


def test(test_loader, net, criterion, device):
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

def load_model(path, model):
    if os.path.isfile(path):
        logging.info("=== loading checkpoint '{}' ===".format(path))

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        logging.info("=== done. ===")

def main():
    global args, config, writer
    writer = SummaryWriter(log_dir=args.work_path + '/event')

    # read config from yaml file
    with open(args.work_path + '/config.yaml') as f:
        config = yaml.load(f)
    # convert to dict
    config = EasyDict(config)
    logger.info(config)

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
        if config.pruning.mode == 'unstructured':
            ckpt_name = "{}_unstruct".format(config.ckpt_name)
        elif config.pruning.mode == 'grid-based':
            ckpt_name = "{}_grid_{}x{}".format(config.ckpt_name, config.pruning.grid_height, config.pruning.grid_width)
        elif config.pruning.mode == 'pattern-based':
            ckpt_name = "{}_pattern_{}x{}".format(config.ckpt_name, config.pruning.size_pattern, config.pruning.num_patterns)
    else:
        ckpt_name = config.ckpt_name
    ckpt_file_name = args.work_path + '/' + ckpt_name + '.pth.tar'
    load_model(ckpt_file_name, net)

    # load training data, do data augmentation and get data loader
    transform_train = transforms.Compose(
        data_augmentation(config))

    transform_test = transforms.Compose(
        data_augmentation(config, is_train=False))

    train_loader, test_loader = get_data_loader(
        transform_train, transform_test, config)

    test(test_loader, net, criterion, device)

    writer.close()


if __name__ == "__main__":
    main()
