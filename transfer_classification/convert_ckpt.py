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
    parser.add_argument('--single_GPU_trained', action='store_true',
                        help='using ema model')
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location="cpu")
    state_dict = ckpt["model"]
    prefix = "encoder.module." if not args.single_GPU_trained else "encoder."

    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        old_k = k
        k = k.replace(prefix, "")
        # if "layer" not in k:
        #     k = "stem." + k
        k = k.replace("shortcut", "downsample")
        k = k.replace("stem.1.", "bn1.")
        k = k.replace("stem.0.", "conv1.")
        print(old_k, "->", k)
        new_state_dict[k] = v.numpy()

    res = {"model": new_state_dict}

    #with open(args.output, "wb") as f:
    #    pkl.dump(res, f)
    torch.save(res, args.output)
