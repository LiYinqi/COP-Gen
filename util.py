from __future__ import print_function

import math
import os
import glob
import random
import pickle
import numpy as np
from PIL import Image

import torch
import torch.optim as optim
from torch.utils.data import Dataset


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, grad_update, class_count, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'grad_updates': grad_update,
        'class_count': class_count,
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def save_linear_model(model, classifier, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'classifier': classifier.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


# GenRep Gaussian Method: https://github.com/ali-design/GenRep/blob/992e571ad1ba94cd40311fe79a0276be13158805/util.py#L404
class GansetDataset(Dataset):
    """The idea is to load the anchor image and its neighbor"""

    def __init__(self, root_dir, neighbor_std=1.0, transform=None, walktype='gaussian', uniformb=None, numcontrast=5, method=None, ratio_data=1):
        """
        Args:
            neighbor_std: std in the z-space
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            walktype: whether we are moving in a gaussian ball or a uniform ball
        """
        super(GansetDataset, self).__init__()
        self.numcontrast = numcontrast
        self.neighbor_std = neighbor_std
        self.uniformb = uniformb
        self.root_dir = root_dir
        self.transform = transform
        self.walktype = walktype
        self.z_dict = dict()
        self.method = method
        self.ratiodata = ratio_data
        self.classes, self.class_to_idx = self._find_classes(self.root_dir)
        self.imglist = []
        
        # get list of anchor images, first check if we have the list in a txt file ow list the images
        extra_rootdir = self.root_dir.replace('indep_20_samples', 'indep_1_samples')
        imgList_filename = os.path.join(extra_rootdir.replace('/train',''), 'ratiodata{}_imgList.txt'.format(self.ratiodata))
        if os.path.isfile(imgList_filename):
            print('Listing images by reading from ', imgList_filename)
            with open(imgList_filename, 'r') as fid:
                self.imglist = fid.readlines()
            self.imglist = [x.rstrip() for x in self.imglist]
        else:
            print("Loading data...")
            self.imglist = glob.glob(os.path.join(extra_rootdir, '*/*_anchor.png'))             # anchor list
            # if self.ratiodata == 1:
            #     max_per_class = 1300
            # else:
            #     max_per_class = int(1300 * self.ratiodata)
            # maks sure we only work on 1300 samples per class (for consistency with imagenet100)
            # indices = [int(x.split('sample')[-1].split('_')[0]) for x in self.imglist]          # anchor image #num_id
            # self.imglist = [imname for imname, ind in zip(self.imglist, indices) if ind < max_per_class]

        if self.ratiodata < 1.:
            # Repeat the dataset to compensate for the lower number of images
            print('Length: {}, will repeat the dataset to compensate for the lower number of images...'.format(len(self.imglist)))
            self.imglist = self.imglist * int(1/self.ratiodata)
        self.dir_size = len(self.imglist)
        print('Length: {}'.format(self.dir_size))

    def _find_classes(self, root_dir):
        classes = glob.glob('{}/*'.format(root_dir))
        print(root_dir)
        classes = [x.split('/')[-1] for x in classes]
        class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
        if self.method == 'SupInv' or self.method == 'UnsupInv':
            # append the z_dataset to the dict:
            for classname in classes:
                with open(os.path.join(self.root_dir, classname, 'z_dataset.pkl'), 'rb') as fid:
                    z_dict = pickle.load(fid)
                self.z_dict[classname] = z_dict
        
        return classes, class_to_idx

    def __len__(self):
        
        return self.dir_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.imglist[idx]
        image = Image.open(img_name)

        if self.numcontrast > 0:
            neighbor = random.randint(1,self.numcontrast)
            img_name_neighbor = self.imglist[idx].replace('anchor','{:.1f}_{}'.format(self.neighbor_std, str(neighbor)))
            if neighbor > 1:
                img_name_neighbor = img_name_neighbor.replace('indep_1_samples', 'indep_20_samples')
            if not os.path.isfile(img_name_neighbor):
                img_name_neighbor = self.imglist[idx].replace('anchor','neighbor_{}'.format( str(neighbor-1)))
        else:
            img_name_neighbor = img_name


        image_neighbor = Image.open(img_name_neighbor)
        label_class = self.imglist[idx].split('/')[-2]
        label = self.class_to_idx[label_class]
        if self.transform:
            oldim = image
            oldimneighbor = image_neighbor
            image = self.transform(image)
            image_neighbor = self.transform(image_neighbor)
            oldim.close()
            oldimneighbor.close()

        z_vect = []
        if self.method == 'SupInv' or self.method == 'UnsupInv': # later can check for Unsupervised inverter will empty labels
            label_dict = self.imglist[idx].split('/')[-2]
            z_vect.append(self.z_dict[label_dict][os.path.basename(img_name)][0]) 
            z_vect.append(self.z_dict[label_dict][os.path.basename(img_name_neighbor)][0])
           # z = np.random.normal(size=128).astype(np.float32)
            # z_vect.append(z)
            # z_vect.append(z)
            return image, image_neighbor, label, label_class, z_vect
        else:
            return image, image_neighbor, label, label_class


# GenRep Steering Method: https://github.com/ali-design/GenRep/blob/992e571ad1ba94cd40311fe79a0276be13158805/util.py#L517
class GansteerDataset(Dataset):
    """The idea is to load the negative-alpha image and its neighbor (positive-alpha)"""

    def __init__(self, root_dir, transform=None, numcontrast=5, method=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        print("Creating dataset: ", root_dir)
        super(GansteerDataset, self).__init__()
        print("Done")
        self.numcontrast = numcontrast
        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(self.root_dir)

        # get list of nalpha images
        self.imglist = glob.glob(os.path.join(self.root_dir, '*/*_anchor.png'))
        print("Loading data...")
        # Make sure there are at most 1300 images per class
        #indices = [int(x.split('sample')[1].split('_')[0]) for x in self.imglist]
        #self.imglist = [imname for imname, ind in zip(self.imglist, indices) if ind < 1300]
        self.dir_size = len(self.imglist)
        print('Length: {}'.format(self.dir_size))

    def _find_classes(self, root_dir):
        classes = glob.glob('{}/*'.format(root_dir))
        classes = [x.split('/')[-1] for x in classes]
        class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
        return classes, class_to_idx

    def __len__(self):

        return self.dir_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.imglist[idx]
        image = Image.open(img_name)
        if self.numcontrast > 0:
            neighbor = random.randint(1,self.numcontrast)
            img_name_neighbor = self.imglist[idx].replace('anchor','neighbor_{}'.format(str(neighbor-1)))
            #if neighbor > 1:
            #    img_name_neighbor = img_name_neighbor.replace('indep_1_samples', 'indep_20_samples')
            #if not os.path.isfile(img_name_neighbor):
            #    img_name_neighbor = self.imglist[idx].replace('anchor','neighbor_{}'.format( str(neighbor-1)))
        else:
            img_name_neighbor = img_name
       # print('anchor, neighbor', img_name, img_name_neighbor)
        image_neighbor = Image.open(img_name_neighbor)
        label = self.imglist[idx].split('/')[-2]
        # with open('./utils/imagenet_class_index.json', 'rb') as fid:
        #     imagenet_class_index_dict = json.load(fid)
        # for key, value in imagenet_class_index_dict.items():
        #     if value[0] == label:
        #         label = key
        #         break
        label = self.class_to_idx[label]
        if self.transform:
            image = self.transform(image)
            image_neighbor = self.transform(image_neighbor)
        return image, image_neighbor, label


# GenRep Random Method (for SupCon only)
class GanRandDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        print("Creating dataset: ", root_dir)
        super(GanRandDataset, self).__init__()
        print("Done")
        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(self.root_dir)

        # get list of nalpha images
        self.imglist = glob.glob(os.path.join(self.root_dir, '*/*_anchor.png'))
        print("Loading data...")
        # Make sure there are at most 1300 images per class
        #indices = [int(x.split('sample')[1].split('_')[0]) for x in self.imglist]
        #self.imglist = [imname for imname, ind in zip(self.imglist, indices) if ind < 1300]
        self.dir_size = len(self.imglist)
        print('Length: {}'.format(self.dir_size))

    def _find_classes(self, root_dir):
        classes = glob.glob('{}/*'.format(root_dir))
        classes = [x.split('/')[-1] for x in classes]
        class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
        return classes, class_to_idx

    def __len__(self):

        return self.dir_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.imglist[idx]
        image = Image.open(img_name)
        img_name_neighbor = self.imglist[idx].replace('_anchor', '')
        img_name_neighbor = img_name_neighbor.replace('seed0', 'seed1')
        image_neighbor = Image.open(img_name_neighbor)
        label = self.imglist[idx].split('/')[-2]
        label = self.class_to_idx[label]
        if self.transform:
            image = self.transform(image)
            image_neighbor = self.transform(image_neighbor)
        return image, image_neighbor, label


# COP-Gen (Load generated contrastive optimal dataset)
class GanSweetDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        print("Creating dataset: ", root_dir)
        super(GanSweetDataset, self).__init__()
        print("Done")

        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(self.root_dir)

        # get list of nalpha images
        self.imglist = glob.glob(os.path.join(self.root_dir, '*/*_anchor.png'))
        print("Loading data...")
        # Make sure there are at most 1300 images per class
        #indices = [int(x.split('sample')[1].split('_')[0]) for x in self.imglist]
        #self.imglist = [imname for imname, ind in zip(self.imglist, indices) if ind < 1300]
        self.dir_size = len(self.imglist)
        print('Length: {}'.format(self.dir_size))

    def _find_classes(self, root_dir):
        classes = glob.glob('{}/*'.format(root_dir))
        classes = [x.split('/')[-1] for x in classes]
        class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
        return classes, class_to_idx

    def __len__(self):

        return self.dir_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.imglist[idx]
        image = Image.open(img_name)
        img_name_neighbor = self.imglist[idx].replace('anchor', 'neighbor')
        image_neighbor = Image.open(img_name_neighbor)
        label = self.imglist[idx].split('/')[-2]
        label = self.class_to_idx[label]
        if self.transform:
            image = self.transform(image)
            image_neighbor = self.transform(image_neighbor)
        return image, image_neighbor, label
