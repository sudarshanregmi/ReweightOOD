#!/bin/bash
# sh scripts/ood/cider/imagenet_test_cider.sh

# available architectures:
# resnet50
# ood
python scripts/eval_ood_imagenet.py \
  --ckpt-path ./results/imagenet_cider_net_reweightood_e30_lr0.001_m_b5.0_c_b4.0_m_w2.0_c_w1.0_default/s0/best.ckpt \
  --arch resnet50 \
  --postprocessor cider \
  --save-score --save-csv #--fsood


# full-spectrum ood
python scripts/eval_ood_imagenet.py \
  --ckpt-path ./results/imagenet_cider_net_reweightood_e30_lr0.001_m_b5.0_c_b4.0_m_w2.0_c_w1.0_default/s0/best.ckpt \
  --arch resnet50 \
  --postprocessor cider \
  --save-score --save-csv --fsood
