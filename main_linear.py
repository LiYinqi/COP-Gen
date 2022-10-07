from __future__ import print_function

import os
import sys
import argparse
import time
import math
import tensorboard_logger as tb_logger

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_linear_model
from networks.resnet_big import SupConResNet, LinearClassifier


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=224, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=32, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=60, help='number of training epochs')
    parser.add_argument('--resume', default='', type=str, help='whether to resume training')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.06, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    parser.add_argument('--warm', action='store_true', help='warm-up for large batch training')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--method', type=str, default='SimCLR', help='method')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['biggan_anchor', 'biggan_random', 'biggan_gauss', 'biggan_steer', 'biggan_sweet',
                                 'bigbigan_anchor', 'bigbigan_gauss', 'bigbigan_steer', 'bigbigan_sweet',
                                 'mnist', 'imagenet100', 'imagenet'], help='dataset')

    # pretrained contrastive encoder
    parser.add_argument('--ckpt', type=str, default='/path/to/trained/contrastive/encoder', help='path to pre-trained model')
    parser.add_argument('--setting1k_CLepoch', type=int, default=100, help="contrastive training epochs, IN1k setting, for distinguishing different encoders")
    parser.add_argument('--desc', type=str, default=None, help='description of the encoder')

    # specifying folders
    parser.add_argument('-d', '--data_folder', type=str, default='./ImageNet1k', help='the folder of the dataset you want to evaluate')
    parser.add_argument('-s', '--save_folder', type=str, default='./Checkpoints', help='the saving folder')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # warm-up for large-batch training,
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if 'ImageNet100' in opt.data_folder:
        opt.tb_path = os.path.join(opt.save_folder, 'linearEva_tensorboards_100')
        opt.model_path = os.path.join(opt.save_folder, 'linearEva_models_100')
    elif 'ImageNet1k' in opt.data_folder:
        opt.tb_path = os.path.join(opt.save_folder, 'linearEva_tensorboards_1k_ptr{}epo'.format(opt.setting1k_CLepoch))
        opt.model_path = os.path.join(opt.save_folder, 'linearEva_models_1k_ptr{}epo'.format(opt.setting1k_CLepoch))
    elif 'MNIST' in opt.data_folder:
        opt.tb_path = os.path.join(opt.save_folder, 'linearEva_tensorboards_mnist')
        opt.model_path = os.path.join(opt.save_folder, 'linearEva_models_mnist')
    else:
        raise ValueError('We use opt.data_folder to distinguish different datasets, {} is not supported'.format(opt.data_folder))

    desc = '{}_{}_lr{}_bs{}_epo{}'.format(opt.method, opt.dataset, opt.learning_rate, opt.batch_size, opt.epochs)
    if opt.desc is not None:
        desc += opt.desc
    opt.tb_folder = os.path.join(opt.tb_path, desc).strip()
    opt.model_folder = os.path.join(opt.model_path, desc).strip()

    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    if 'gan' in opt.dataset or 'imagenet' in opt.dataset:
        opt.img_size = 128
        opt.img_dim = 3
        opt.n_cls = 1000
    elif 'mnist' in opt.dataset:
        opt.img_size = 28
        opt.img_dim = 1
        opt.n_cls = 10
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_loader(opt):
    # construct data loader
    if 'gan' in opt.dataset or 'imagenet' in opt.dataset:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normalize = transforms.Normalize(mean=mean, std=std)
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(int(opt.img_size), scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize(int(opt.img_size / 0.875)),       # 128/0.875
            transforms.CenterCrop(int(opt.img_size)),           # 128
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder, 'train'),
                                             transform=train_transform)
        val_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder, 'val'),
                                           transform=val_transform)

    elif 'mnist' in opt.dataset:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = datasets.MNIST(root=opt.data_folder,
                                       transform=transform)
        val_dataset = datasets.MNIST(root=opt.data_folder, train=False,
                                     transform=transform)

    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader, val_loader


def set_model(opt):
    model = SupConResNet(name=opt.model, img_size=opt.img_size, in_channel=opt.img_dim)
    criterion = torch.nn.CrossEntropyLoss()
    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
            # NOTE: uncomment the following codes for loading 1 GPU trained model
            # new_state_dict = {}
            # for k, v in state_dict.items():
            #     if "encoder.module" not in k:
            #         k = k.replace("encoder", "encoder.module")
            #         new_state_dict[k] = v
            # state_dict = new_state_dict
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
        model.load_state_dict(state_dict,  strict=False)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)
            top5.update(acc5[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print(' * Acc@5 {top5.avg:.3f}'.format(top5=top5))
    return losses.avg, top1.avg, top5.avg


def main():
    best_acc = 0
    best_acc_corr5 = 0
    opt = parse_option()
    print(opt)

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    print("Logging in {}".format(opt.tb_folder))

    # training routine
    init_epoch = 1
    if len(opt.resume) > 0:
        model_ckp = torch.load(opt.resume)
        init_epoch = model_ckp['epoch'] + 1
        model.load_state_dict(model_ckp['model'])
        classifier.load_state_dict(model_ckp['classifier'])
        optimizer.load_state_dict(model_ckp['optimizer'])

    for epoch in range(init_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))
        logger.log_value('loss_train', loss, epoch)
        logger.log_value('acc_train', acc, epoch)

        # eval for one epoch
        loss, val_acc1, val_acc5 = validate(val_loader, model, classifier, criterion, opt)
        logger.log_value('loss_eval', loss, epoch)
        logger.log_value('acc_eva', val_acc1, epoch)
        logger.log_value('acc5_eva;', val_acc5, epoch)
        if val_acc1 > best_acc:
            best_acc = val_acc1
            best_acc_corr5 = val_acc5

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(opt.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_linear_model(model, classifier, optimizer, opt, epoch, save_file)

    print('best accuracy: {:.2f}, corresponding top5_acc: {:.2f}'.format(best_acc, best_acc_corr5))
    fn = os.path.join(opt.tb_folder, "result_linear.txt")
    with open(fn, 'w+') as f:
        f.write('best accuracy: {:.2f}, corresponding top5_acc: {:.2f}'.format(best_acc, best_acc_corr5))


if __name__ == '__main__':
    main()
