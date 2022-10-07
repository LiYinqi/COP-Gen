import os
import sys
import time
import argparse
import numpy as np
from PIL import Image
import tensorboard_logger as tb_logger

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torchvision import transforms

# GAN related
from pytorch_pretrained_gans import make_gan
from scipy.stats import truncnorm

sys.path.append('../')
from networks.resnet_big import SupConResNet
from losses import SupConLoss

dim_z = 120


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--num_samples', type=int, default=1600000, help='number of training samples')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--img_size', type=int, default=128, help='img_size')
    parser.add_argument('--truncation', type=float, default=1.0, help='truncation of BigBiGAN')
    parser.add_argument('--save_freq', type=int, default=1000, help='frequency (num_samples) to save weights')
    parser.add_argument('--save_folder', type=str, default='./walk_weights_bigbigan', help='saving folder')

    # optimization
    parser.add_argument('--lr_walk', type=float, default=0.00001, help='learning rate of navigator')
    parser.add_argument('--lr_MI', type=float, default=0.00003, help='learning rate of mutual_info_estimator')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--temp', type=float, default=0.1, help='temperature for InfoNCE loss')

    # model
    parser.add_argument('--backbone', type=str, default='resnet18',
                        help='arch of mutual information estimator')
    parser.add_argument('--walker_type', type=str, default='nonlinear', choices=['nonlinear', 'linear'],
                        help='linear z+Wz or nonlinear z+NN(z) navigation in latent space')
    parser.add_argument('--reduction_ratio', type=float, default=1.0,
                        help='reduction ratio of the nonlinear MLP walker')

    # data space transformation tx
    parser.add_argument('--simclr_aug', action='store_true', help='use full data augs')
    parser.add_argument('--removeCrop', action='store_true', help='remove random crop from data augs')
    parser.add_argument('--removeColor', action='store_true', help='remove random color from data augs')

    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')

    opt = parser.parse_args()

    desc = 'w_COPGen_SimCLR_{}_{}Samples_bsz{}_imgsz{}_lrWalk{}_lrMI{}_beta1={}_beta2={}_temp{}_reduRat{}_L2={}_trial{}'.format(
        opt.backbone, opt.num_samples, opt.batch_size, opt.img_size, opt.lr_walk, opt.lr_MI, opt.beta1, opt.beta2, opt.temp, opt.reduction_ratio, opt.weight_decay, opt.trial)

    if opt.simclr_aug:
        desc += '_simclrAug'
    elif opt.removeCrop:
        desc += '_removeCrop'
    elif opt.removeColor:
        desc += '_removeColor'
    else:
        desc += '_NOsimclrAug'

    if opt.walker_type == 'nonlinear':
        desc += '_z+NN(z)'
    elif opt.walker_type == 'linear':
        desc += '_z+W2W1z'
    else:
        raise ValueError('walker type not supported: {}'.format(opt.walker_type))

    opt.ckpt_folder = os.path.join(opt.save_folder, 'ckpts', desc)
    opt.tb_folder = os.path.join(opt.save_folder, 'tensorboard', desc)

    if not os.path.isdir(opt.ckpt_folder):
        os.makedirs(opt.ckpt_folder)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    print(opt)

    return opt


def truncated_noise_sample(batch_size=1, dim_z=dim_z, truncation=1., seed=None):
    """ Create a truncated noise vector.
        Params:
            batch_size: batch size.
            dim_z: dimension of z
            truncation: truncation value to use
            seed: seed for the random generator
        Output:
            array of shape (batch_size, dim_z)
    """
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation * values


class NonlinearWalk(nn.Module):
    def __init__(self, dim_z, reduction_ratio=1.0):
        super(NonlinearWalk, self).__init__()
        self.walker = nn.Sequential(
            nn.Linear(dim_z, int(dim_z / reduction_ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(int(dim_z / reduction_ratio), dim_z)
        )
        # weight initialization: default

    def forward(self, input):
        return self.walker(input)


class LinearWalk(nn.Module):
    def __init__(self, dim_z, reduction_ratio=1.0):
        super(LinearWalk, self).__init__()
        self.walker = nn.Sequential(
            nn.Linear(dim_z, int(dim_z / reduction_ratio)),
            nn.Linear(int(dim_z / reduction_ratio), dim_z)
        )
        # weight initialization: default

    def forward(self, input):
        return self.walker(input)


def set_models(opt):
    # trained IGM
    G = make_gan(gan_type='bigbigan').eval()

    # walker (navigator)
    if opt.walker_type == 'nonlinear':
        nn_walker = NonlinearWalk(dim_z, opt.reduction_ratio)
    elif opt.walker_type == 'linear':
        nn_walker = LinearWalk(dim_z, opt.reduction_ratio)
    else:
        raise ValueError('walker type not supported: {}'.format(opt.walker_type))

    # mutual info estimator
    MI_estimator = SupConResNet(name=opt.backbone, img_size=int(opt.img_size))

    if torch.cuda.device_count() > 1:
        print('Using {} gpus for models'.format(torch.cuda.device_count()))
        G = torch.nn.DataParallel(G)
        nn_walker = torch.nn.DataParallel(nn_walker)
        MI_estimator.encoder = torch.nn.DataParallel(MI_estimator.encoder)

    G.to('cuda')
    nn_walker = nn_walker.to('cuda')
    MI_estimator = MI_estimator.to('cuda')

    return G, nn_walker, MI_estimator


def transform(imgs, out_size, opt):
    '''
    :param imgs: [N, 3, H, W]
    :return: [N, 3, H', W']
    '''
    imgs = (imgs + 1) / 2.0     # values: (-1, 1) to (0, 1)
    imgs_resized = torch.zeros([imgs.shape[0], 3, int(out_size), int(out_size)])

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if opt.simclr_aug:
        img_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=int(out_size), scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.Normalize(mean=mean, std=std),
        ])
    elif opt.removeCrop:
        img_transform = transforms.Compose([
            transforms.Resize(int(out_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.Normalize(mean=mean, std=std),
        ])
    elif opt.removeColor:
        img_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=int(out_size), scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        img_transform = transforms.Compose([
            transforms.Resize(int(out_size)),
            transforms.Normalize(mean=mean, std=std),
        ])

    # imgs in the batch should have random (different) transformations
    for b in range(imgs.shape[0]):
        imgs_resized[b] = img_transform(imgs[b])

    return imgs_resized


def train(opt):
    G, nn_walker, MI_estimator = set_models(opt)

    # criterion and optimizer
    criterion = SupConLoss(temperature=opt.temp).to('cuda')

    optimizer_walk = optim.Adam(nn_walker.parameters(), lr=opt.lr_walk,
                                betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    optimizer_MI = optim.Adam(MI_estimator.parameters(), lr=opt.lr_MI,
                              betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)

    cudnn.benchmark = True

    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    opt.save_point = opt.save_freq

    img_save_path = os.path.join(opt.ckpt_folder, 'images')
    if not os.path.isdir(img_save_path):
        os.makedirs(img_save_path)

    loss_MI_values = []
    loss_walker_values = []

    # train loop
    for batch_idx in range(opt.num_samples // opt.batch_size):
        start_time = time.time()

        # sample z randomly
        z = truncated_noise_sample(truncation=opt.truncation, batch_size=opt.batch_size, seed=None)    # [N, 120]
        z = torch.from_numpy(z).to('cuda')

        # generate anchor image
        with torch.no_grad():
            img_ori = G(z)                                      # [N, 3, 128, 128]  values: (-1, 1)

        # z' = z + Tz(z)
        z_new = z + nn_walker(z)

        # positives
        img_new = G(z_new)

        # data space transformations tx
        img_ori_tx = transform(img_ori, opt.img_size, opt)      # [N, 3, 128, 128]  values: (0, 1)
        img_new_tx = transform(img_new, opt.img_size, opt)

        # cat anchors and postives
        images = torch.cat([img_ori_tx.unsqueeze(1), img_new_tx.unsqueeze(1)], dim=1)
        images = images.view(-1, 3, int(opt.img_size), int(opt.img_size)).cuda(non_blocking=True)

        ###### Update mutual_info_estimator: min InfoNCE loss (max lower bound of Mutual Info) ######
        optimizer_MI.zero_grad()

        features = MI_estimator(images.detach())
        features = features.view(opt.batch_size, 2, -1)

        loss_MI = criterion(features)

        loss_MI.backward()
        optimizer_MI.step()
        #############################################################################################

        ###### Update navigator (nn_walker): max InfoNCE loss (min Mutual Info) #####################
        optimizer_walk.zero_grad()

        features = MI_estimator(images)
        features = features.view(opt.batch_size, 2, -1)

        loss_walker = - criterion(features)

        loss_walker.backward()
        optimizer_walk.step()
        #############################################################################################

        elapsed_time = time.time() - start_time

        loss_MI_values.append(loss_MI.detach().cpu().numpy())
        loss_walker_values.append(loss_walker.detach().cpu().numpy())

        logger.log_value('loss_MI', loss_MI, batch_idx * opt.batch_size)
        logger.log_value('loss_walker', loss_walker, batch_idx * opt.batch_size)

        logger.log_value('z_new_mean', z_new.mean().detach().cpu().numpy(), batch_idx * opt.batch_size)
        logger.log_value('z_new_std', z_new.std().detach().cpu().numpy(), batch_idx * opt.batch_size)
        logger.log_value('z_new-z_mean', (z_new-z).mean().detach().cpu().numpy(), batch_idx * opt.batch_size)
        logger.log_value('z_new-z_std', (z_new-z).std().detach().cpu().numpy(), batch_idx * opt.batch_size)

        print('Time:%.2f, trained_samples:%d, loss_MI:%.4f, loss_walker:%.4f, z_new_std: %.4f, z_new-z_std: %.4f' % (
            elapsed_time, batch_idx * opt.batch_size, loss_MI, loss_walker, z_new.std(), (z_new-z).std()))

        # save anchor-positive pairs for monitoring the minimax training process
        if ((batch_idx * opt.batch_size) >= opt.save_point) and (batch_idx > 0):
            grid_images = vutils.make_grid(torch.cat((img_ori, img_new.detach())), nrow=opt.batch_size, normalize=True)
            grid_images = (grid_images.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            Image.fromarray(grid_images).resize((grid_images.shape[1], grid_images.shape[0])).save(
                f'{img_save_path}/image_Nsamples{batch_idx * opt.batch_size}.png', 'PNG')

            print('Saving navigator. #trained_samples = {}'.format(batch_idx * opt.batch_size))
            torch.save({'nn_walker': nn_walker}, os.path.join(opt.ckpt_folder, 'w_COPGen_{}.pth'.format(batch_idx * opt.batch_size)))
            
            opt.save_point += opt.save_freq

    np.save(os.path.join(opt.ckpt_folder, 'loss_MI_values.npy'), loss_MI_values)
    np.save(os.path.join(opt.ckpt_folder, 'loss_walker_values.npy'), loss_walker_values)


if __name__ == '__main__':
    opt = parse_option()
    train(opt)
