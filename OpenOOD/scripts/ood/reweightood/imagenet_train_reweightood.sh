#!/bin/bash
# sh scripts/ood/cider/imagenet_train_cider.sh

python main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/networks/cider_net.yml \
    configs/pipelines/train/train_reweightood.yml \
    configs/preprocessors/base_preprocessor.yml \
    --preprocessor.name cider \
    --network.backbone.name resnet50 \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
    --optimizer.lr 0.001 \
    --optimizer.num_epochs 30 \
    --dataset.train.batch_size 512 \
    --trainer.trainer_args.m_b 5 \
    --trainer.trainer_args.c_b 4 \
    --trainer.trainer_args.m_w 2 \
    --trainer.trainer_args.c_w 1 \
    --num_gpus 1 --num_workers 16 \
    --merge_option merge \
    --seed 0
