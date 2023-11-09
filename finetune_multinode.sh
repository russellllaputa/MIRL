# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export PADDLE_TRAINERS_NUM=2
MIM_PATH=/root/paddlejob/workspace/env_run/huangguoxi/PLSC/task/ssl/tmae/
PLSC_PATH=/root/paddlejob/workspace/env_run/huangguoxi/PLSC/
export PYTHONPATH=$PLSC_PATH:$MIM_PATH:$PYTHONPATH

unset PADDLE_TRAINER_ENDPOINTS
export PADDLE_NNODES=2
export PADDLE_MASTER="10.127.27.21:6070"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PADDLE_JOB_ID=TMAE_FINETUNE
export DISTRIBUTED_TRAINER_ENDPOINTS="10.127.27.21:6070,10.127.12.11:6070"

# for single node finetune
# batch_size 32, ACCUM_ITER=4, effective batch size: 1024
# batch_size 128, ACCUM_ITER=1, effective batch size: 1024

# 4 nodes finetune setting
ACCUM_ITER=1
BATCH_SIZE=128
MODEL=vit_base_patch16
RESUME=/root/paddlejob/workspace/env_run/huangguoxi/tmae_finetune/$MODEL/checkpoint-76.pd
PRETRAIN_CHKPT=/root/paddlejob/workspace/env_run/huangguoxi/tmae_output/tmae_vit_base_patch16/checkpoint-299.pd
IMAGENET_DIR=/root/paddlejob/workspace/env_run/ILSVRC2012/
OUTPUT_DIR=/root/paddlejob/workspace/env_run/huangguoxi/tmae_finetune/$MODEL
python3 -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    --log_dir="/root/paddlejob/workspace/env_run/log" \
    main_finetune.py \
    --accum_iter $ACCUM_ITER \
    --batch_size $BATCH_SIZE \
    --model $MODEL \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --blr 7.5e-4 --layer_decay 0.6 \
    --warmup_epochs 20 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${OUTPUT_DIR} \
    --seed 888 \
    --use_ema \
    --dist_eval --data_path ${IMAGENET_DIR} 
    # --resume $RESUME

