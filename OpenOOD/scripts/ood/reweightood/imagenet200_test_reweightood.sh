#!/bin/bash
# sh scripts/ood/reweightood/imagenet200_test_reweightood.sh

# ood
python scripts/eval_ood.py \
   --id-data imagenet200 \
   --root ./results/imagenet200_cider_net_reweightood_e90_lr0.1_m_b5.0_c_b4.0_m_w2.0_c_w1.0_default \
   --postprocessor cider \
   --save-score --save-csv #--fsood

# full-spectrum ood
python scripts/eval_ood.py \
   --id-data imagenet200 \
   --root ./results/imagenet200_cider_net_reweightood_e90_lr0.1_m_b5.0_c_b4.0_m_w2.0_c_w1.0_default \
   --postprocessor cider \
   --save-score --save-csv --fsood
