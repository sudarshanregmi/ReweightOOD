#!/bin/bash
# sh scripts/ood/reweightood/imagenet200_train_reweightood.sh

python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/networks/cider_net.yml \
    configs/pipelines/train/train_reweightood.yml \
    configs/preprocessors/base_preprocessor.yml \
    --preprocessor.name cider \
    --network.backbone.name resnet18_224x224 \
    --optimizer.lr 0.1 \
    --optimizer.num_epochs 90 \
    --dataset.train.batch_size 512 \
    --trainer.trainer_args.m_b 5 \
    --trainer.trainer_args.c_b 4 \
    --trainer.trainer_args.m_w 2 \
    --trainer.trainer_args.c_w 1 \
    --num_gpus 1 --num_workers 16 \
    --merge_option merge \
    --seed $RANDOM
