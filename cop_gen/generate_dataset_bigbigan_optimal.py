import os
import tqdm
import argparse
import json
import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image

# GAN related
from pytorch_pretrained_gans import make_gan
from scipy.stats import truncnorm


def convert_to_images(obj):
    if not isinstance(obj, np.ndarray):
        obj = obj.detach().numpy()
    obj = obj.transpose((0, 2, 3, 1))
    obj = np.clip(((obj + 1) / 2.0) * 256, 0, 255)
    img = []
    for i, out in enumerate(obj):
        out_array = np.asarray(np.uint8(out), dtype=np.uint8)
        img.append(Image.fromarray(out_array))
    return img


def truncated_noise_sample(batch_size=1, dim_z=120, truncation=1., seed=None):
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


def sample(opt):
    with open(opt.json_file, 'rb') as fid:
        imagenet_class_index_dict = json.load(fid)
    imagenet_class_index_keys = list(imagenet_class_index_dict.keys())

    print('Loading the model ...')
    G = make_gan(gan_type='bigbigan').eval()
    if torch.cuda.device_count() > 1:
        print('Using {} gpus for G'.format(torch.cuda.device_count()))
        G = torch.nn.DataParallel(G)
    G.to('cuda')

    # load pretrained navigator
    walk_ckpt = torch.load(opt.walker_path)
    nn_walker = walk_ckpt['nn_walker']
    nn_walker.device_ids = [0]
    nn_walker = nn_walker.to('cuda')

    for key in tqdm(imagenet_class_index_keys):
        class_dir_name = os.path.join(opt.output_path, opt.partition, imagenet_class_index_dict[key][0])
        if os.path.isdir(class_dir_name):
            continue
        os.makedirs(class_dir_name, exist_ok=True)
        idx = int(key)
        z_dict = dict()
        print('Generating images for class {}'.format(idx))

        seed = opt.start_seed + idx
        noise_vectors = truncated_noise_sample(truncation=opt.truncation, batch_size=opt.num_imgs, seed=seed)
        noise_vectors = torch.from_numpy(noise_vectors).to('cuda')

        for batch_start in range(0, opt.num_imgs, opt.batch_size):
            s = slice(batch_start, min(opt.num_imgs, batch_start + opt.batch_size))
            z = noise_vectors[s]
            bsz = z.shape[0]

            with torch.no_grad():
                # get anchors
                out_anchors = G(z)

                # get neighbors (positives)
                z_new = z + nn_walker(z)
                out_positives = G(z_new)

            ims_anchors = convert_to_images(out_anchors.cpu())
            ims_positives = convert_to_images(out_positives.cpu())

            # save anchor and its neighbors
            for b in range(bsz):
                im = ims_anchors[b]
                im_name = 'seed%04d_sample%05d_anchor.%s' % (seed, batch_start+b, opt.imformat)
                im.save(os.path.join(class_dir_name, im_name))
                z_dict[im_name] = [z[b].cpu().numpy(), idx]

                im = ims_positives[b]
                im_name = 'seed%04d_sample%05d_neighbor.%s' % (seed, batch_start+b, opt.imformat)
                im.save(os.path.join(class_dir_name, im_name))
                z_dict[im_name] = [z_new[b].detach().cpu().numpy(), idx]

        with open(os.path.join(class_dir_name, 'z_dataset.pkl'), 'wb') as fid:
            pickle.dump(z_dict, fid)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate dataset using bigbigan and navigator")
    parser.add_argument('--dataset', default='1k', choices=['1k', '100'], type=str)
    parser.add_argument('--out_dir', default='../GeneratedImgs', type=str)
    parser.add_argument('--partition', default='train', type=str)
    parser.add_argument('--truncation', default=1.0, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_imgs', default=1300, type=int)
    parser.add_argument('--imformat', default='png', type=str)
    parser.add_argument('--start_seed', default=0, type=int)
    parser.add_argument('--walker_path', default='/path/to/pretrained/navigator', type=str)
    parser.add_argument('--desc', default='description', type=str, help='this will be the tag of this specfic dataset, added to the end of the dataset name')
    opt = parser.parse_args()

    if opt.dataset == '1k':
        opt.json_file = './imagenet1k_class_index.json'
        opt.output_path = (os.path.join(opt.out_dir, '1k_bigbigan128tr{}_{}'.format(opt.truncation, opt.desc)))

    elif opt.dataset == '100':
        opt.json_file = './imagenet100_class_index.json'
        opt.output_path = (os.path.join(opt.out_dir, '100_bigbigan128tr{}_{}'.format(opt.truncation, opt.desc)))

    else:
        raise ValueError('Invalid dataset')

    print(opt)

    sample(opt)
