#!/bin/bash
# sh scripts/ood/reweightood/cifar100_train_reweightood.sh

python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/networks/cider_net.yml \
    configs/pipelines/train/train_reweightood.yml \
    configs/preprocessors/base_preprocessor.yml \
    --preprocessor.name cider \
    --network.backbone.name resnet18_32x32 \
    --dataset.train.batch_size 512 \
    --trainer.trainer_args.m_b 5 \
    --trainer.trainer_args.c_b 4 \
    --trainer.trainer_args.m_w 2 \
    --trainer.trainer_args.c_w 1 \
    --num_workers 8 \
    --optimizer.num_epochs 100 \
    --seed $RANDOM
