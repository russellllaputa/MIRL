# [NeurIPS 2023] Masked Image Residual Learning for Scaling Deeper Vision Transformers
Official implementation of the paper [Masked Image Residual Learning for Scaling Deeper Vision Transformers](https://arxiv.org/abs/2309.14136)
> Abstract: Deeper Vision Transformers (ViTs) are more challenging to train. We expose a
degradation problem in deeper layers of ViT when using masked image modeling
(MIM) for pre-training. To ease the training of deeper ViTs, we introduce a self-supervised learning framework called Masked Image Residual Learning (MIRL),
which significantly alleviates the degradation problem, making scaling ViT along
depth a promising direction for performance upgrade. We reformulate the pretraining objective for deeper layers of ViT as learning to recover the residual of the
masked image. We provide extensive empirical evidence showing that deeper ViTs
can be effectively optimized using MIRL and easily gain accuracy from increased
depth. With the same level of computational complexity as ViT-Base and ViT-Large,
we instantiate 4.5× and 2× deeper ViTs, dubbed ViT-S-54 and ViT-B-48. The
deeper ViT-S-54, costing 3× less than ViT-Large, achieves performance on par with
ViT-Large. ViT-B-48 achieves 86.2% top-1 accuracy on ImageNet. On one hand,
deeper ViTs pre-trained with MIRL exhibit excellent generalization capabilities
on downstream tasks, such as object detection and semantic segmentation. On the
other hand, MIRL demonstrates high pre-training efficiency. With less pre-training
time, MIRL yields competitive performance compared to other approaches.

<div align='center'>
<img src="asset/mirl_arch.png" alt="Architecture" width="880" style="display: block;"/>
</div>




Code and pretrained models will be uploaded soon


## Updates

***07/Oct/2023***

The preprint version is public at [arxiv](https://arxiv.org/abs/2309.14136).


## Pretrain on ImageNet-1K
The following table provides pretrained checkpoints and logs used in the paper.
| model | pre-trained 300 epochs | pre-trained 800 epochs  |
| :---: | :---: | :---: |
| ViT-B-48 | [checkpoint](https://pan.baidu.com/s/1H3gpMl4-S0gFibv5xbJDHQ?pwd=mirl), [log](https://pan.baidu.com/s/1ZmW1KyrzLBvD52buX1GqAQ?pwd=mirl)| [checkpoint](https://pan.baidu.com/share/init?surl=bpyLctZy6Ww2QQ-s9KiSjQ&pwd=mirl), [log](https://pan.baidu.com/s/1XKBXSLbyPqeXFREaTdmupQ?pwd=mirl)|
| ViT-S-54 | -| [checkpoint](https://pan.baidu.com/share/init?surl=oF0Gnhhlx6gdgUhIjjDl6Q&pwd=mirl), log 


