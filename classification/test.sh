#!/bin/sh
EXP_NAME=exp1

CUDA_VISIBLE_DEVICES=0  python ./scripts/test.py \
    --img_dir=/data/VOCdevkit/VOC2012/JPEGImages/ \
    --test_list=./data/train_cls.txt \
    --arch=vgg \
    --batch_size=1 \
    --dataset=pascal_voc \
    --input_size=256 \
	  --num_classes=20 \
    --restore_from=./runs/${EXP_NAME}/model/pascal_voc_epoch_14.pth \
    --save_dir=./runs/${EXP_NAME}/attention/ \