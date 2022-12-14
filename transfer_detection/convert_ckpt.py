# disclaimer: inspired by MoCo official repo.

import pickle as pkl
import torch
import pdb
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert Models')
    parser.add_argument('input', metavar='I',
                        help='input model path')
    parser.add_argument('output', metavar='O',
                        help='output path')
    parser.add_argument('--ema', action='store_true',
                        help='using ema model')
    parser.add_argument('--single_GPU_trained', action='store_true',
                        help='using ema model')
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location="cpu")
    if args.ema:
        state_dict = ckpt["model_ema"]
        prefix = "encoder."
    else:
        state_dict = ckpt["model"]
        prefix = "encoder.module." if not args.single_GPU_trained else "encoder."

    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        old_k = k
        k = k.replace(prefix, "")
        if "layer" not in k:
            k = "stem." + k
        k = k.replace("layer1", "res2")
        k = k.replace("layer2", "res3")
        k = k.replace("layer3", "res4")
        k = k.replace("layer4", "res5")
        k = k.replace("bn1", "conv1.norm")
        k = k.replace("bn2", "conv2.norm")
        k = k.replace("bn3", "conv3.norm")
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        k = k.replace("shortcut.0", "shortcut")
        k = k.replace("shortcut.1", "shortcut.norm")
        k = k.replace("stem.1.", "conv1.norm.")
        k = k.replace("stem.1", "conv1")
        k = k.replace("stem.0.", "conv1.")
        print(old_k, "->", k)
        new_state_dict[k] = v.numpy()

    res = {"model": new_state_dict,
           "__author__": "Xavier",
           "matching_heuristics": True}

    #with open(args.output, "wb") as f:
    #    pkl.dump(res, f)
    torch.save(res, args.output)
