## Fine-tuning Pre-trained MIRL for Classification


To fine-tune with **multi-node distributed training**, run the following on 4 nodes with 8 GPUs each:
```

# for single node finetune
# batch_size 32, ACCUM_ITER=4, effective batch size: 1024
# batch_size 128, ACCUM_ITER=1, effective batch size: 1024

# 4 nodes finetune setting
ACCUM_ITER=4
BATCH_SIZE=16
MODEL=vit_base_depth48_patch16

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
    --epochs 60 \
    --blr 7.5e-4 --layer_decay 0.88 \
    --warmup_epochs 30 \
    --weight_decay 0.05 --drop_path 0. --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${OUTPUT_DIR} \
    --seed 888 \
    --use_ema \
    --dist_eval --data_path ${IMAGENET_DIR}


- Here the effective batch size is 32 (`batch_size` per gpu) * 4 (`nodes`) * 8 (gpus per node) = 1024.
- `blr` is the base learning rate. The actual `lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `lr` = `blr` * effective batch size / 256.
- Here we use EMA to improve the results of our long-tail tasks in practical applications, but we clarify this will not yield great different results on ImageNet-1K.
  The hyperparameter settings without EAM can be found from https://github.com/facebookresearch/mae/blob/main/FINETUNE.md

