#!/bin/bash

GPUS=3      # for multiple gpus, separted by space ' '

GUIDANCE=neural_codes       # can be `neural_codes`, `geom_disp` or `both`
#GUIDANCE=geom_disp       
#GUIDANCE=both       

python train.py \
    --gpus $GPUS \
    --model_version model \
    --config "TRAIN.STEPS_PER_EPOCH=30000;TRAIN.BATCH_SIZE=16;MODEL.GUIDANCE=$GUIDANCE" \
    --train_dir ./train_log/$GUIDANCE \
    --init_ckpt_file checkpoints/PWM_$GUIDANCE.pth