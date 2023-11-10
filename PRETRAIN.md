## Pre-training MIRL

To pre-train ViT-B-48 (recommended default) with **multi-node distributed training**, run the following on 8 nodes with 8 GPUs each:
```


# If you use a single node
# batch_size 64, ACCUM_ITER=8, effective batch size: 4096
# batch_size 256, ACCUM_ITER=2, effective batch size: 4096

MODEL=mirl_vit_base_depth48_patch16

python3 -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    --log_dir=${LOG_DIR} \
    main_pretrain.py \
    --accum_iter $ACCUM_ITER \
    --batch_size $BATCH_SIZE \
    --model $MODEL \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 801 \
    --warmup_epochs 40 \
    --num_workers 10 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${OUTPUT_DIR} \


```
- Here the effective batch size is 64 (`batch_size` per gpu) * 8 (`nodes`) * 8 (gpus per node) = 4096. If memory or # gpus is limited, use `--accum_iter` to maintain the effective batch size, which is `batch_size` (per gpu) * `nodes` * 8 (gpus per node) * `accum_iter`.
- `blr` is the base learning rate. The actual `lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `lr` = `blr` * effective batch size / 256.
- Here `--norm_pix_loss` no longer effects
