# Official Pytorch Implementation of DToP

### Dynamic Token Pruning in Plain Vision Transformers for Semantic Segmentation 
Quan Tang, Bowen Zhang, Jiajun Liu, Fagui Liu, Yifan Liu

ICCV 2023. [[arxiv]](https://arxiv.org/abs/2308.01045)

This repository contains the official Pytorch implementation of training & evaluation code and the pretrained models for DToP

As shown in the following figure, the network is naturally split into stages using inherent auxiliary blocks.

<img src="./resources/fig-1-1.png">

## Highlights
* **Dynamic Token Pruning** We introduce a dynamic token pruning paradigm based on the early exit of easy-to-recognize tokens for semantic segmentation transformers.
* **Controllable prune ratio** One hyperparameter to control the trade-off between computation cost and accuracy.
* **Generally applicable** e apply DToP to mainstream semantic segmentation transformers and can reduce up to 35% computational cost without a notable accuracy drop.

## Getting started 
1. requirements
```
torch==2.0.0 mmcls==1.0.0.rc5, mmcv==2.0.0 mmengine==0.7.0 mmsegmentation==1.0.0rc6 
```
or up-to-date mmxx series till 9 Aug 2023

## Training
To aquire the base model
```
python tools dist_train.sh config/prune/BASE_segvit_ade20k_large.py  $work_dirs$
```
To prune on the base model
```
python tools dist_train_load.sh  config/prune/prune_segvit_ade20k_large.py  $work_dirs$  $path_to_ckpt$
```

## Eval
```
python tools/dist_test.sh  config/prune/prune_segvit_ade20k_large.py  $path_to_ckpt$
```

## Datasets
Please follow the instructions of [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) data preparation

## Results
### Ade20k
| Method       | Backbone  | mIoU | GFlops | config | ckpt |
|--------------|-----------|------|--------|--------|------|
| Segvit       | Vit-base  | 49.6 | 109.9  | [config](./config/prune/BASE_segvit_ade20k.py)       |      |
| Segvit-prune |  Vit-base | 49.8 |   86.8 | [config](./config/prune/prune_segvit_ade20k.py)       |      |
| Segvit       | Vit-large | 53.3 |  617.0 | [config](./config/prune/BASE_segvit_ade20k_large.py)       |      |
| Segvit-prune | Vit-large | 52.8 |  412.8 |  [config](./config/prune/prune_segvit_ade20k_large.py)      |      |

### Pascal Context
| Method       | Backbone  | mIoU | GFlops | config | ckpt |
|--------------|-----------|------|--------|--------|------|
| Segvit       | Vit-large | 63.0 |  315.4 | [config](./config/prune/BASE_segvit_pc.py)       |      |
| Segvit-prune | Vit-large | 62.7 |  224.3 | [config](./config/prune/prune_segvit_pc.py)       |      |

### COCO-Stuff-10K
| Method       | Backbone  | mIoU | GFlops | config | ckpt |
|--------------|-----------|------|--------|--------|------|
| Segvit       | Vit-large | 47.4 |  366.9 | [config](./config/prune/BASE_segvit_cocostuff10k.py)       |      |
| Segvit-prune | Vit-large | 47.1 |  276.2 | [config](./config/prune/prune_segvit_cocostuff10k.py)       |      |



## License
For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors.

## Citation
