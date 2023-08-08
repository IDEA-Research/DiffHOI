#!/usr/bin/env bash

set -x
export SD_Config="/path/to/v1-inference.yaml"
export SD_ckpt="/path/to/v1-5-pruned-emaonly.ckpt"
EXP_DIR=exps/diffhoi_l_hico

python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env \
        main.py \
        --pretrained params/dino_swinl.pth \
        --output_dir ${EXP_DIR} \
        --model_name diffhoi_l \
        -c configs/DiffHOI_L.py \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone swin_L_384_22k \
        --num_queries 900 \
        --dec_layers 3 \
        --epochs 90 \
        --lr_drop 60 \
        --use_nms_filter \
        --ft_clip_with_small_lr \
        --with_clip_label \
        --with_obj_clip_label
