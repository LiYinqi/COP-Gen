# COP-Gen

This repo is the official code of COP-Gen, for reproducing the results of our paper.


## Requirements

- Linux
- Python 3.8
- PyTorch 1.7.1


## Navigator Training
Run the following command to train the navigator in pretrained BigBiGAN's latent space.
```bash
cd cop_gen
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_navigator_bigbigan_optimal.py --simclr_aug --batch_size BATCH_SIZE
```
where: 
- `simclr_aug` is set to use SimCLR data space augmentation by default.
- `BATCH_SIZE` is set as 176 in the paper, which is the largest value that can be achieved on 4 NVIDIA 2080 Ti GPUs.

Beolow gives the training losses of the navigator and the mutual infomation estimator.

<img src='figures/minimax_loss.png' width=700>

We also monitor the mean and std of the navigator during training as an auxiliary monitor signal for terminating the training process,
as shown in the figure below.

<img src='figures/Tz_mean_std.png' width=700>

A simple and effective way is to watch the qualtity of generated positive pairs, 
which will be saved at `cop_gen/walk_weights_bigbigan/ckpts/name_of_experiment/images`.

### Note
Since the final performance is affected by the choice of navigator checkpoint 
(i.e., when to terminate the training process),
we here provide our [trained navigator checkpoint](./PretrainedModels/navigator_bigbigan128/w_COPGen_pretained.pth) for the following experiments, 
to reproduce quantitative and qualitative results in the paper.


## Dataset Generation
After training the navigator, we use it together of the pretrained GAN to generate contrastive pretraining dataset offline.
```bash
cd cop_gen
CUDA_VISIBLE_DEVICES=0 python generate_dataset_bigbigan_optimal.py --dataset DATASET_TYPE --out_dir OUT_DIR --walker_path /path/to/pretained/navigator
```
where: 
- `DATASET_TYPE` is set to `1k` (`100`) for ImageNet-1K (ImageNet-100) scale pretraining dataset.
- `OUT_DIR` is the path to save the generated dataset.


## Contrastive Learning
Run the following command to perform contrastive learning on the generated dataset.
```bash
CUDA_VISIBLE_DEVICES=0,1 python main_unified.py \
  --method=SimCLR \
  --dataset=bigbigan_sweet \
  --data_folder=/path/to/generated/dataset \
  --save_folder=/path/to/save/contrastive/encoders \
  --batch_size=BATCH_SIZE \
  --epochs=100 \
  --learning_rate=LR \
  --cosine
```
where: 
- `BATCH_SIZE` is set as 224 in the paper, which is the largest value that can be achieved on 2 NVIDIA 2080 Ti GPUs.
- `LR` is set as `0.03 * BATCH_SIZE / 256 = 0.02625` in the paper, following [GenRep](https://github.com/ali-design/GenRep).

We provide our trained contrastive encoder at [Google Drive](https://drive.google.com/drive/folders/13hJqYfyGEk3FFcrVCCPyWvkcc9LLIlhB?usp=sharing),
for quickly evaluating downstream performances.


## Linear Evaluation
Run the following command to train a linear classifier on top of the trained encoder.
```bash
CUDA_VISIBLE_DEVICES=0,1 python main_linear.py \
  --ckpt=/path/to/trained/contrastive/encoders \
  --data_folder=/path/to/ImageNet1k \
  --dataset=bigbigan_sweet \
  --save_folder=/path/to/save/linear/models \
  --batch_size=BATCH_SIZE \
  --epochs=60 \
  --learning_rate=LR \
  --cosine
```
where: 
- `BATCH_SIZE` is set as 224 to align with the pretraining stage.
- `LR` is set as `2 * BATCH_SIZE / 256 = 1.75` in the paper, following [GenRep](https://github.com/ali-design/GenRep).


## Transfer Detection
Follow the steps below to evaluate the pretrained contrastive encoder on Pascal VOC.
1. Install [detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
2. Convert the pretrained encoder to detectron2's format:
```bash
cd transfer_detection
python convert_ckpt.py /path/to/trained/encoder /path/to/converted/ckpt
```
3. Put VOC2007 and VOC2012 datasets to `./transfer_detection/datasets` directory.
4. Run training:
```bash
cd transfer_detection
CUDA_VISIBLE_DEVICES=0,1 python train_net.py \
  --num-gpus 2 \
  --config-file ./config/pascal_voc_R_50_C4_transfer.yaml \
  MODEL.WEIGHTS /path/to/converted/ckpt \
  OUTPUT_DIR /path/to/outputs \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.BASE_LR 0.005 \
  INPUT.MAX_SIZE_TRAIN 1024 \
  INPUT.MAX_SIZE_TEST 1024 \
  SOLVER.MAX_ITER 96000 \
  SOLVER.STEPS 72000,88000 \
  SOLVER.WARMUP_ITERS 400 \
  DATALOADER.NUM_WORKERS 8
```


## Transfer Classification and Semi-Supervised Learning
Follow the steps below to evaluate the pretrained contrastive encoder on transfer classification and semi-supervised learning tasks.
1. Install [VISSL](https://vissl.readthedocs.io/en/v0.1.6/installation.html) and you will get a `vissl` folder under your `/home/user` directory.
2. Convert the pretrained encoder to VISSL's format:
```bash
cd transfer_classification
python convert_ckpt.py /path/to/trained/encoder /path/to/converted/ckpt
```
3. Preparing datasets using the scripts and guidelines from [vissl/extra_scripts](https://github.com/facebookresearch/vissl/tree/main/extra_scripts).
4. Run training:
```bash
# This is an example for linear transfer classification on Caltech-101, 
# based on the documentation: https://vissl.readthedocs.io/en/v0.1.6/evaluations/linear_benchmark.html
cd vissl
CUDA_VISIBLE_DEVICES=0 python tools/run_distributed_engines.py \
  config=benchmark/linear_image_classification/caltech101/eval_resnet_8gpu_transfer_caltech101_linear \
  config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=model \
  config.MODEL.WEIGHTS_INIT.APPEND_PREFIX="trunk.base_model._feature_blocks." \
  config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/path/to/converted/ckpt \
  config.CHECKPOINT.DIR=/path/to/outputs \
  config.DATA.TRAIN.DATA_PATHS=["datasets/caltech101/train"] \
  config.DATA.TEST.DATA_PATHS=["datasets/caltech101/test"] \
  config.MODEL.SYNC_BN_CONFIG.GROUP_SIZE=-1 \
  config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
  config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=256 \
  config.DATA.TEST.BATCHSIZE_PER_REPLICA=256 \
  config.DATA.NUM_DATALOADER_WORKERS=8
  
# This is an example for semi-supervised learning on 1% labeled ImageNet,
# based on the documentation: https://vissl.readthedocs.io/en/v0.1.6/evaluations/semi_supervised.html
cd vissl
CUDA_VISIBLE_DEVICES=0 python tools/run_distributed_engines.py \
  config=benchmark/semi_supervised/imagenet1k/eval_resnet_8gpu_transfer_in1k_semi_sup_fulltune_per01 \
  +config/benchmark/semi_supervised/imagenet1k/dataset=simclr_in1k_per01 \
  config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=model \
  config.MODEL.WEIGHTS_INIT.APPEND_PREFIX="trunk._feature_blocks." \
  config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/path/to/converted/ckpt \
  config.CHECKPOINT.DIR=/path/to/outputs \
  config.DATA.TRAIN.DATA_PATHS=["datasets/imagenet1k_semisupv_1percent/train_images.npy"] \
  config.DATA.TRAIN.LABEL_PATHS=["datasets/imagenet1k_semisupv_1percent/train_labels.npy"] \
  config.DATA.TEST.DATA_SOURCES=[disk_filelist] \
  config.DATA.TEST.LABEL_SOURCES=[disk_filelist] \
  config.DATA.TEST.DATA_PATHS=["datasets/imagenet1k_npy/val_images.npy"] \
  config.DATA.TEST.LABEL_PATHS=["datasets/imagenet1k_npy/val_labels.npy"] \
  config.MODEL.SYNC_BN_CONFIG.GROUP_SIZE=-1 \
  config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
  config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=256 \
  config.DATA.TEST.BATCHSIZE_PER_REPLICA=256 \
  config.DATA.NUM_DATALOADER_WORKERS=8
```


## Acknowledgments
This code is based on the following repositories:
- [GenRep](https://github.com/ali-design/GenRep)
- [PyTorch Pretrained GANs](https://github.com/lukemelas/pytorch-pretrained-gans)

