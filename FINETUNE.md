## Fine-tuning Pre-trained MIRL for Classification


Script for ViT-B-48
```


# 4 nodes finetune setting
ACCUM_ITER=4
BATCH_SIZE=16
MODEL=vit_base_depth48_patch16

python3 -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    --log_dir=${LOG_DIR} \
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
```



Script for ViT-S-54
```

# 2 nodes finetune setting
ACCUM_ITER=2
BATCH_SIZE=64
MODEL=vit_small_depth54_patch16

python3 -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    --log_dir=${LOG_DIR} \
    main_finetune.py \
    --accum_iter $ACCUM_ITER \
    --batch_size $BATCH_SIZE \
    --model $MODEL \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --blr 7.5e-4 --layer_decay 0.88 \
    --warmup_epochs 20 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${OUTPUT_DIR} \
    --seed 888 \
    --use_ema \
    --dist_eval --data_path ${IMAGENET_DIR}

```

- Here the effective batch size is (`batch_size` per gpu) * (`nodes`) * (gpus per node) = 2048.
- `blr` is the base learning rate. The actual `lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `lr` = `blr` * effective batch size / 256.
