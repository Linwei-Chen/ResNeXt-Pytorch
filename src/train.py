# -*- coding: utf-8 -*-
from __future__ import division

""" 
Trains a ResNeXt Model on Cifar10 and Cifar 100. Implementation as defined in:
Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.

"""

__author__ = "charles"
__email__ = "charleschen2013@163.com"

import argparse
import os
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from model import CifarResNeXt
import time
from lr_scheduler import WarmUpMultiStepLR
from logger import ModelSaver, Logger
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader


def config():
    parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('--data_path', type=str, default='/Users/chenlinwei/dataset/cifar-10-batches-py',
                        help='Root for the Cifar dataset.')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='Choose between Cifar10/100.')
    # Optimization options
    parser.add_argument('--optimizer', '-op', type=str, default='adam', help='Optimizer to train model.')
    parser.add_argument('--epochs', '-e', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=2, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='The Learning Rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--test_bs', type=int, default=10)
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default=None, help='Folder to save checkpoints.')
    parser.add_argument('--save_steps', '-ss', type=int, default=200, help='steps to save checkpoints.')
    parser.add_argument('--load', '-l', type=str, help='Checkpoint path to resume / test.')
    parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
    # Architecture
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--cardinality', type=int, default=16, help='Model cardinality (group).')
    parser.add_argument('--base_width', type=int, default=64, help='Number of channels in each group.')
    parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=12, help='Pre-fetching threads.')
    # i/o
    parser.add_argument('--log', type=str, default=None, help='Log folder.')
    args = parser.parse_args()
    if args.save is None:
        args.save = f'../{args.dataset}'
    if args.log is None:
        args.log = f'../{args.dataset}'
    args.scheduler_name = f'{args.optimizer}_scheduler'
    return args


def get_cifar_transform():
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])
    return train_transform, test_transform


def get_data_set(args, train_transform, test_transform):
    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
        nlabels = 10
    else:
        train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        nlabels = 100

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.prefetch, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                             num_workers=args.prefetch, pin_memory=True)
    return nlabels, train_loader, test_loader


def get_model(args):
    # Init model, criterion, and optimizer
    net = CifarResNeXt(args.cardinality, args.depth, nlabels, args.base_width, args.widen_factor)
    print(net)
    return net


def get_optimizer(args, model):
    op_dict = {
        'adam': optim.Adam(model.parameters(), lr=args.lr),
        'sgd': optim.SGD(net.parameters(), args.lr, momentum=args.momentum,
                         weight_decay=args.decay, nesterov=True)
    }
    assert isinstance(args.optimizer, str)
    assert args.optimizer.lower() in [op.lower() for op in op_dict.keys()]
    return op_dict.get(args.optimizer)


def model_accelerate(args, net):
    if args.ngpu > 1 and torch.cuda.is_available():
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0 and torch.cuda.is_available():
        net.cuda()
    return net


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# train function (forward, backward, update)
def train(args, net, optimizer, train_loader, logger, model_saver):
    clock = time.perf_counter()
    net.train()
    loss_avg = 0.0
    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # forward
        output = net(data)

        # backward
        optimizer.zero_grad()
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        print(f'idx:{batch_idx} | loss:{loss.item():.8f} | lr:{get_lr(optimizer)} | time:{time.perf_counter() - clock}')
        clock = time.perf_counter()

        # exponential moving average
        # loss_avg = loss_avg * 0.2 + float(loss) * 0.8
        # logger.log(key='train_loss', data=loss_avg)
        logger.log(key='train_loss', data=loss.item())
        if batch_idx % args.save_steps == 0:
            logger.visualize(key='train_loss', range=(-1000, -1))
            logger.save_log()
            model_saver.save(name='model', model=net)
            model_saver.save(name=args.optimizer, model=optimizer)
            # break


# test function (forward only)
def test(net, test_loader, logger, model_saver):
    net.eval()
    loss_avg = 0.0
    correct = 0
    for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
        data, target = data.cuda(), target.cuda()

        # forward
        output = net(data)
        loss = F.cross_entropy(output, target)

        # accuracy
        pred = output.data.max(1)[1]
        correct += float(pred.eq(target.data).sum())

        # test loss average
        loss_avg += float(loss)
        # if batch_idx > 100:break

    logger.log(key='test_loss', data=loss_avg / len(test_loader))

    test_accuracy = correct / len(test_loader.dataset)
    if logger.get_data(key='test_accuracy') == [] or test_accuracy > max(logger.get_data(key='test_accuracy')):
        model_saver.save(name=f'model_{test_accuracy}', model=net)
    logger.log(key='test_accuracy', data=test_accuracy)


if __name__ == '__main__':
    args = config()

    train_transform, test_transform = get_cifar_transform()
    nlabels, train_loader, test_loader = get_data_set(args,
                                                      train_transform=train_transform,
                                                      test_transform=test_transform)

    # Init checkpoints
    if not os.path.isdir(args.save):
        os.makedirs(args.save)

    model_saver = ModelSaver(save_path=args.save, name_list=['model', 'scheduler', args.optimizer])
    net = get_model(args)
    model_saver.load(name='model', model=net)

    net = model_accelerate(args, net)
    optimizer = get_optimizer(args, model=net)
    model_saver.load(name=args.optimizer, model=optimizer)

    scheduler = WarmUpMultiStepLR(optimizer=optimizer, milestones=args.schedule, warm_up_iters=0)
    model_saver.load(name=args.scheduler_name, model=scheduler)

    logger = Logger(save_path=args.log, json_name='log')
    # Main loop
    best_accuracy = 0.0
    for _ in range(args.epochs):
        _, train_loader, test_loader = get_data_set(args,
                                                    train_transform=train_transform,
                                                    test_transform=test_transform)
        scheduler.step()
        epoch_now = scheduler.state_dict()['last_epoch']
        print(f'*** Epoch now:{epoch_now}.')
        if epoch_now >= args.epochs:
            print('*** Training finished!')
        train(args, net, optimizer, train_loader, logger, model_saver)
        test(net, test_loader, logger, model_saver)

        model_saver.save(name=args.scheduler_name, model=scheduler)
        logger.save_log()
        logger.visualize()
