#!/bin/sh
EXP_NAME=naive_cam_visualization

CUDA_VISIBLE_DEVICES=1 python3 ./scripts/train.py \
    --img_dir=../../dataset/VOC2012/JPEGImages/ \
    --train_list=./data/train_cls.txt \
    --test_list=./data/val_cls.txt \
    --epoch=15 \
    --lr=0.001 \
    --batch_size=5 \
    --dataset=pascal_voc \
    --input_size=256 \
	  --disp_interval=100 \
	  --num_classes=20 \
	  --num_workers=8 \
	  --snapshot_dir=./runs/${EXP_NAME}/model/  \
    --att_dir=./runs/${EXP_NAME}/accu_att/ \
    --decay_points='5,10' \
    --wandb_name=${EXP_NAME}