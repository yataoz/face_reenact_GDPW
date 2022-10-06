#!/bin/bash

GPU=3

SRC_IMG=data/face_edit/obama.jpg
INTERACTIVE=1

# ========== choose the model that you wish to run by updating the GUIDANCE
GUIDANCE=neural_codes       # can be `neural_codes`, `geom_disp` or `both`
#GUIDANCE=geom_disp       
#GUIDANCE=both       

# PWM + neural_codes
python face_edit_demo.py \
    --gpus $GPU \
    --config "MODEL.GUIDANCE=$GUIDANCE" \
    --src_img $SRC_IMG \
    --interactive $INTERACTIVE \
    --init_ckpt_file checkpoints/PWM_$GUIDANCE.pth

