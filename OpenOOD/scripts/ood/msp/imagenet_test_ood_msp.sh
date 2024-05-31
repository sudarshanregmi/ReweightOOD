#!/bin/bash
# sh scripts/ood/msp/imagenet_test_ood_msp.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/datasets/imagenet/imagenet_ood.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/msp.yml \
    --num_workers 10 \
    --ood_dataset.image_size 256 \
    --dataset.test.batch_size 256 \
    --dataset.val.batch_size 256 \
    --network.pretrained True \
    --network.checkpoint 'results/pretrained_weights/resnet50_imagenet1k_v1.pth' \
    --merge_option merge

############################################
# we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood_imagenet.py

# available architectures:
# resnet50, swin-t, vit-b-16
# ood
python scripts/eval_ood_imagenet.py \
   --tvs-pretrained \
   --arch resnet50 \
   --postprocessor msp \
   --save-score --save-csv #--fsood


# full-spectrum ood
python scripts/eval_ood_imagenet.py \
   --tvs-pretrained \
   --arch resnet50 \
   --postprocessor msp \
   --save-score --save-csv --fsood
