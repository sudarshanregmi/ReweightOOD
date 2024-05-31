#!/bin/bash
# sh scripts/ood/reweightood/cifar10_test_reweightood.sh

python scripts/eval_ood.py \
   --id-data cifar10 \
   --root ./results/cifar10_cider_net_reweightood_e100_lr0.1_m_b5.0_c_b4.0_m_w2.0_c_w1.0_default \
   --postprocessor cider \
   --save-score --save-csv
