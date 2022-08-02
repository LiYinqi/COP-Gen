from __future__ import print_function

import numpy as np
import os
import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
import tensorboard_logger as tb_logger

from util import TwoCropTransform, AverageMeter, GansetDataset, GansteerDataset, GanRandDataset, GanSweetDataset
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss
import oyaml as yaml
import json


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=224, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--resume', default='', type=str, help='whether to resume training')
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    parser.add_argument('--warm', action='store_true', help='warm-up for large batch training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # model and dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['biggan_anchor', 'biggan_random', 'biggan_gauss', 'biggan_steer', 'biggan_sweet',
                                 'bigbigan_anchor', 'bigbigan_gauss', 'bigbigan_steer', 'bigbigan_sweet',
                                 'mnist', 'cifar10', 'cifar100', 'imagenet100', 'imagenet'], help='dataset')

    # method
    parser.add_argument('--ratiodata', type=float, default=1.0, help='ratio of the data')
    parser.add_argument('--numcontrast', type=int, default=1, help='num of neighbors')
    parser.add_argument('--method', type=str, default='SimCLR', choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument('--removeimtf', action='store_true', help='whether to remove all simclr transforms, but remain center crop')
    parser.add_argument('--removeCrop', action='store_true', help='whether to remove RRC from simclr transforms')
    parser.add_argument('--removeColor', action='store_true', help='whether to remove color augs from simclr transforms')
    parser.add_argument('--remainCropOnly', action='store_true', help='whether to remove simclr transforms, and remain random resized crop only')

    # temperature
    parser.add_argument('--temp', type=float, default=0.1, help='temperature for loss function')

    # specifying folders
    parser.add_argument('-d', '--data_folder', type=str, default='./ImageNet1k', help='the data folder')
    parser.add_argument('-s', '--save_folder', type=str, default='./Checkpoints', help='the saving folder')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = opt.data_folder
    opt.model_path = os.path.join(opt.save_folder, '{}/{}_models'.format(opt.method, opt.dataset))
    opt.tb_path = os.path.join(opt.save_folder, '{}/{}_tensorboard'.format(opt.method, opt.dataset))

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_ncontrast{}_ratiodata{}_lr{}_decay{}_bsz{}_temp{}_trial{}_CLepoch{}'. \
        format(opt.method, opt.dataset, opt.model, opt.numcontrast, opt.ratiodata, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial, opt.epochs)

    if opt.removeimtf:
        opt.model_name = '{}_noimtf'.format(opt.model_name)

    if opt.removeCrop:
        opt.model_name = '{}_noCrop'.format(opt.model_name)

    if opt.removeColor:
        opt.model_name = '{}_noColor'.format(opt.model_name)

    if opt.remainCropOnly:
        opt.model_name = '{}_onlyRRCimtf'.format(opt.model_name)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    opt.model_name = '{}_{}'.format(opt.model_name, os.path.basename(opt.data_folder))

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if 'biggan' in opt.dataset or 'bigbigan' in opt.dataset or 'imagenet' in opt.dataset:
        opt.img_size = 128
        opt.img_dim = 3
        opt.n_cls = 1000
    elif opt.dataset == 'cifar10' or 'cifar10_' in opt.dataset:
        opt.img_size = 32
        opt.img_dim = 3
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.img_size = 32
        opt.img_dim = 3
        opt.n_cls = 100
    elif 'mnist' in opt.dataset:
        opt.img_size = 28
        opt.img_dim = 1
        opt.n_cls = 10
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10' or 'cifar10_' in opt.dataset:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif 'biggan' in opt.dataset or 'bigbigan' in opt.dataset or 'imagenet' in opt.dataset:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'mnist' in opt.dataset:
        mean = None
        std = None
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)
    opt.mean = mean
    opt.std = std

    if opt.removeimtf:
        train_transform = transforms.Compose([
            transforms.CenterCrop(size=int(opt.img_size*0.875)),
            transforms.Resize(size=int(opt.img_size)),
            transforms.ToTensor(),
            normalize,
        ])
    elif opt.removeCrop:
        train_transform = transforms.Compose([
            transforms.CenterCrop(size=int(opt.img_size * 0.875)),
            transforms.Resize(size=int(opt.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    elif opt.removeColor:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=int(opt.img_size), scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    elif opt.remainCropOnly:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=int(opt.img_size), scale=(0.2, 1.)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=int(opt.img_size), scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    if 'mnist' in opt.dataset:
        train_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10,
                                    translate=[0.1, 0.1],
                                    scale=[0.9, 1.1]),
            transforms.ToTensor()
        ])

    if opt.dataset == 'mnist':
        train_dataset = datasets.MNIST(root=opt.data_folder,
                                       transform=TwoCropTransform(train_transform))

    elif opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform))

    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform))

    elif opt.dataset == 'imagenet100' or opt.dataset == 'imagenet' or 'anchor' in opt.dataset:
        train_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder, 'train'),
                                             transform=TwoCropTransform(train_transform))

    elif 'gauss' in opt.dataset:
        train_dataset = GansetDataset(root_dir=os.path.join(opt.data_folder, 'train'),
                                      transform=train_transform, numcontrast=opt.numcontrast, ratio_data=opt.ratiodata)

    elif 'steer' in opt.dataset:
        train_dataset = GansteerDataset(root_dir=os.path.join(opt.data_folder, 'train'),
                                        transform=train_transform, numcontrast=opt.numcontrast)

    elif 'sweet' in opt.dataset:
        train_dataset = GanSweetDataset(root_dir=os.path.join(opt.data_folder, 'train'),
                                        transform=train_transform)

    elif opt.dataset == 'biggan_random':
        train_dataset = GanRandDataset(root_dir=os.path.join(opt.data_folder, 'train'),
                                      transform=train_transform)

    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
    model = SupConResNet(name=opt.model, img_size=int(opt.img_size), in_channel=opt.img_dim)
    criterion = SupConLoss(temperature=opt.temp)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt, grad_update, class_count, logger):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    top1 = AverageMeter()
    end = time.time()

    print("Start train")

    ############# Comments from GenRep, we set ratiodata = 1 in our experiments #############
    # Size_dataset always should be 1.3M for imagenet1K and 130K for imagenet100
    # the ratiodata means how many unique images we have compared to the original dataset
    # if ratiodata < 1, we just repeat the dataset in the dataloader 1 / ratiodata times.
    # When ratiodata > 1, we train for 1 / ratiodata times, to keep the number of grad updates constant

    # iter_epoch is how many iterations every epoch has, so it will be len(data)/batch_size for ratio < 1
    # and len(data/ratio) / batch_size for the above reason
    #########################################################################################

    size_dataset = len(train_loader.dataset) / max(opt.ratiodata, 1)

    # how many iterations per epoch
    iter_epoch = int(size_dataset / opt.batch_size)
    for idx, data in enumerate(train_loader):
        grad_update += 1
        if idx % iter_epoch == 0:
            losses.reset()
            curr_epoch = int(epoch + (idx / iter_epoch))
            adjust_learning_rate(opt, optimizer, curr_epoch)

        if len(data) == 2:
            images = data[0]
            labels = data[1]
        elif len(data) == 3:
            images = data[:2]
            labels = data[2]
        elif len(data) == 4:
            images = data[:2]
            labels = data[2]
            labels_class = data[3]
        else:
            raise NotImplementedError

        data_time.update(time.time() - end)

        images = torch.cat([images[0].unsqueeze(1), images[1].unsqueeze(1)], dim=1)

        images = images.view(-1, opt.img_dim, int(opt.img_size), int(opt.img_size)).cuda(non_blocking=True)

        labels_np = [x for x in labels.numpy()]
        for x in labels_np:
            class_count[x] += 1

        labels = labels.cuda(non_blocking=True)

        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        features = features.view(bsz, 2, -1)
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                  epoch, idx + 1, len(train_loader), batch_time=batch_time,
                  data_time=data_time, loss=losses))
            sys.stdout.flush()

        if (idx + 1) % iter_epoch == 0 or (epoch == 1 and idx == 0):
            if idx == 0 and epoch == 1:
                curr_epoch = 0
            else:
                curr_epoch = int(epoch + (idx / iter_epoch))

            # tensorboard logger
            logger.log_value('loss_avg', losses.avg, curr_epoch)
            logger.log_value('grad_update', grad_update, curr_epoch)
            logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], curr_epoch)

    return losses.avg, grad_update, class_count


def main():
    opt = parse_option()

    with open(os.path.join(opt.save_folder, 'optE.yml'), 'w') as f:
        yaml.dump(vars(opt), f, default_flow_style=False)
    print(opt)

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    init_epoch = 1
    if len(opt.resume) > 0:
        model_ckp = torch.load(opt.resume)
        state_dict = model_ckp['model']

        # For loading 1 gpu trained ckpt on >=2 gpus ONLY
        # new_state_dict = {}
        # update = False
        # for k, v in state_dict.items():
        #     if "encoder.module" not in k:
        #         k = k.replace("encoder", "encoder.module")
        #         new_state_dict[k] = v
        #         update = True
        # if update:
        #   state_dict = new_state_dict

        if not torch.cuda.device_count() > 1:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict

        init_epoch = model_ckp['epoch'] + 1
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(model_ckp['optimizer'])

    skip_epoch = 1
    if opt.ratiodata > 1:
        skip_epoch = int(opt.ratiodata)

    grad_update = 0
    class_count = np.zeros(1000)
    for epoch in range(init_epoch, opt.epochs + 1, skip_epoch):

        # train for one epoch
        time1 = time.time()
        loss, grad_update, class_count = train(train_loader, model, criterion, optimizer, epoch, opt, grad_update, class_count, logger)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if opt.ratiodata <= 1:
            if epoch % opt.save_freq == 0 or epoch == 1:
                save_file = os.path.join(
                    opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                save_model(model, optimizer, opt, epoch, grad_update, class_count, save_file)
        else:
            if epoch % opt.save_freq == 1:
                save_file = os.path.join(
                    opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                save_model(model, optimizer, opt, epoch, grad_update, class_count, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, grad_update, class_count, save_file)


if __name__ == '__main__':
    main()
