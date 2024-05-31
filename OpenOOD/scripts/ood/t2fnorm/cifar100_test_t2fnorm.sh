#!/bin/bash

python scripts/eval_ood.py \
   --id-data cifar100 \
   --root ./results/cifar100_resnet18_32x32_t2fnorm_e100_lr0.1_alpha0.1_default \
   --postprocessor t2fnorm \
   --save-score --save-csv
