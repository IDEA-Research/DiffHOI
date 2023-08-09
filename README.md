# **DiffHOI: Boosting Human-Object Interaction Detection with Text-to-Image Diffusion Model**
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boosting-human-object-interaction-detection/zero-shot-human-object-interaction-detection)](https://paperswithcode.com/sota/zero-shot-human-object-interaction-detection?p=boosting-human-object-interaction-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boosting-human-object-interaction-detection/human-object-interaction-detection-on-hico)](https://paperswithcode.com/sota/human-object-interaction-detection-on-hico?p=boosting-human-object-interaction-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boosting-human-object-interaction-detection/human-object-interaction-detection-on-v-coco)](https://paperswithcode.com/sota/human-object-interaction-detection-on-v-coco?p=boosting-human-object-interaction-detection)
### [Project Page](https://diffhoi.github.io/) | [Paper](https://arxiv.org/pdf/2305.12252.pdf) | Data (Coming Soon)

### SynHOI dataset Visiualization
<img src="assets/SynHOI_vis.gif" style="height:550px" />  



## 🔥 Key Features
- ### **DiffHOI**: The first framework leverages the generative and representative capabilities to benefit the HOI task.
- ### **SynHOI dataset**: A class-balance, large-scale, and high-diversity synthetic HOI dataset.

⚔️ We are dedicated to enhancing and expanding the SynHOI dataset. We will release it soon, together with more powerful models for HICO-DET and V-COCO through SynHOI-Pretraining.


## 🐟 Installation
Installl the dependencies.
```
pip install -r requirements.txt
```
Clone and build CLIP.
```
git clone https://github.com/openai/CLIP.git && cd CLIP && python setup.py develop && cd ..
```

Compiling CUDA operators for deformable attention.
```
cd models/DiffHOI_L/ops
python setup.py build install
cd ../../..
```

Download the checkpoint of [Stable-Diffusion](https://github.com/runwayml/stable-diffusion) (we use v1-5 by default). Please also follow its instructions to install the required packages. 

## 🦈 Data preparation

### HICO-DET
HICO-DET dataset can be downloaded [here](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk). After finishing downloading, unpack the tarball (`hico_20160224_det.tar.gz`) to the `data` directory.

Instead of using the original annotations files, we use the annotation files provided by the PPDM authors. The annotation files can be downloaded from [here](https://drive.google.com/open?id=1WI-gsNLS-t0Kh8TVki1wXqc3y2Ow1f2R). The downloaded annotation files have to be placed as follows.
```
data
 └─ hico_20160224_det
     |─ annotations
     |   |─ trainval_hico.json
     |   |─ test_hico.json
     |   └─ corre_hico.npy
     :
```

### V-COCO
First clone the repository of V-COCO from [here](https://github.com/s-gupta/v-coco), and then follow the instruction to generate the file `instances_vcoco_all_2014.json`. Next, download the prior file `prior.pickle` from [here](https://drive.google.com/drive/folders/10uuzvMUCVVv95-xAZg5KS94QXm7QXZW4). Place the files and make directories as follows.
```
DiffHOI
 |─ data
 │   └─ v-coco
 |       |─ data
 |       |   |─ instances_vcoco_all_2014.json
 |       |   :
 |       |─ prior.pickle
 |       |─ images
 |       |   |─ train2014
 |       |   |   |─ COCO_train2014_000000000009.jpg
 |       |   |   :
 |       |   └─ val2014
 |       |       |─ COCO_val2014_000000000042.jpg
 |       |       :
 |       |─ annotations
 :       :
```
The annotation file have to be converted to the HOIA format. The conversion can be conducted as follows.
```
PYTHONPATH=data/v-coco \
        python convert_vcoco_annotations.py \
        --load_path data/v-coco/data \
        --prior_path data/v-coco/prior.pickle \
        --save_path data/v-coco/annotations
```
Note that only Python2 can be used for this conversion because `vsrl_utils.py` in the v-coco repository shows a error with Python3.

V-COCO annotations with the HOIA format, `corre_vcoco.npy`, `test_vcoco.json`, and `trainval_vcoco.json` will be generated to `annotations` directory.



## 🚢 Pre-trained model
Download the pretrained model of DETR detector for [ResNet50](https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth), and put it to the `params` directory.
```
python ./tools/convert_parameters.py \
        --load_path params/detr-r50-e632da11.pth \
        --save_path params/detr-r50-pre-2branch-hico.pth \
        --num_queries 64

python ./tools/convert_parameters.py \
        --load_path params/detr-r50-e632da11.pth \
        --save_path params/detr-r50-pre-2branch-vcoco.pth \
        --dataset vcoco \
        --num_queries 64
```

Download the pretrained model of Deformable DETR detector for [Swin-L](https://drive.google.com/drive/folders/1qD5m1NmK0kjE5hh-G17XUX751WsEG-h_), and put it to the `params` directory.

## 🚀 Results and Models
### 😎 DiffHOI on HICO-DET.
|                    | Full (D) | Rare (D) | Non-rare (D) | Full(KO) | Rare (KO) | Non-rare (KO) |                                           Download                                            |             Conifg             |
|:-------------------|:--------:|:--------:|:------------:|:--------:|:---------:|:-------------:|:---------------------------------------------------------------------------------------------:|:------------------------------:|
| DiffHOI-S (R50)    |  34.41   |  31.07   |    35.40     |  37.31   |   34.56   |     38.14     | [model](https://drive.google.com/drive/folders/1WUkiE4meJ3r0F8YnV7bYQzdDoUnGw_LT?usp=sharing) | [config](configs/DiffHOI_S.py) |
| DiffHOI-L (Swin-L) |  40.63   |  38.10   |    41.38     |  43.14   |   40.24   |     44.01     |                                           [model](https://drive.google.com/drive/folders/1WUkiE4meJ3r0F8YnV7bYQzdDoUnGw_LT?usp=sharing)                                           | [config](configs/DiffHOI_L.py) |

## ⭐ Training
After the preparation, you can start training with the following commands.
### HICO-DET
```
sh ./run/hico_s.sh
```

### V-COCO
```
sh ./run/vcoco_s.sh
```
### Zero-shot
```
sh ./run/hico_s_zs_nf_uc.sh
```

## ⭐ Testing
### HICO-DET
```
sh ./run/hico_s_eval.sh
```
```
sh ./run/hico_l_eval.sh
```
## Citation
Please consider citing our paper if it helps your research.
```
@article{yang2023boosting,
          title={Boosting Human-Object Interaction Detection with Text-to-Image Diffusion Model},
          author={Yang, Jie and Li, Bingliang and Yang, Fengyu and Zeng, Ailing and Zhang, Lei and Zhang, Ruimao},
          journal={arXiv preprint arXiv:2305.12252},
          year={2023}
        }
```

## Acknowledge
This repo is mainly based on [GEN-VLKT](https://github.com/YueLiao/gen-vlkt) Licensed under MIT Copyright (c) [2022] [Yue Liao] , [DINO](https://github.com/IDEA-Research/DINO) under Apache 2.0 Copyright (c) [2022] [IDEA-Research]. We thank their well-organized code!
