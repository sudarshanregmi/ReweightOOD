#!/bin/bash

python scripts/eval_ood_imagenet.py \
  --ckpt-path ./results/imagenet_resnet50_t2fnorm_e30_lr0.001_alpha0.1_default/s0/best.ckpt \
  --arch resnet50 \
  --postprocessor t2fnorm \
  --save-score --save-csv
