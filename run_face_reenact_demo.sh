#!/bin/bash

GPU=3

# Step 1
# ========== uncomment the task you wish to run, and specify src image and driving video

# same id face reenactment
#SRC_IMG=data/sample_test/id10283/vaK4t1-WD4M/016892#017089/0000080_img.jpg
#DRIVING_VIDEO=data/sample_test/id10283/vaK4t1-WD4M/010562#010679

## cross id face reenactment
SRC_IMG=data/sample_test/id10283/vaK4t1-WD4M/010562#010679/0000000_img.jpg
DRIVING_VIDEO=data/sample_test/id10280/NXjT3732Ekg/001093#001192

#SRC_IMG=data/sample_test/id10280/NXjT3732Ekg/001093#001192/0000000_img.jpg
#DRIVING_VIDEO=data/sample_test/id10283/vaK4t1-WD4M/010562#010679

# Step 2
# ========== choose the model that you wish to run by updating the GUIDANCE
GUIDANCE=neural_codes       # can be `neural_codes`, `geom_disp` or `both`
#GUIDANCE=geom_disp       
#GUIDANCE=both       

python face_reenact_demo.py \
    --gpus $GPU \
    --config "MODEL.GUIDANCE=$GUIDANCE" \
    --src_img $SRC_IMG \
    --driving_video $DRIVING_VIDEO \
    --init_ckpt_file checkpoints/PWM_$GUIDANCE.pth