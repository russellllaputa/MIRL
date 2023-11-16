from functools import partial
import paddle
import paddle.nn as nn
from models.vision_transformer import PatchEmbed, Block, CrossBlock
from util.pos_embed import get_2d_sincos_pos_embed
from nn.init import constant_, normal_, uniform_, xavier_uniform_, zeros_
# from models.vgg import vgg16
from util.functions import patchify, unpatchify, mask_unpatchify

class MIRLViT(nn.Layer):
    """ Masked Image Residual Learning with VisionTransformer backbone
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=1024,
                 depth=24,
                 num_heads=16,
                 decoder_embed_dims=[512, 512, 512, 512],
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
                 norm_pix_loss=False,
                 out_indices=[5, 7],
                 queue_size=8192,
                 temperature=0.07,
                 weight_norm_pred=False,
                 ):
        super().__init__()
        self.out_indices = out_indices
        self.queue_size = queue_size
        self.temperature = temperature
        self.weight_norm_pred = weight_norm_pred
        # --------------------------------------------------------------------------
        # Encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        # self.patch_embed.proj.weight.stop_gradient = True
        # self.patch_embed.proj.bias.stop_gradient = True
        # for param in self.patch_embed.parameters():
        #     param.stop_gradient = True
        num_patches = self.patch_embed.num_patches
        self.cls_token = self.create_parameter(shape=(1, 1, embed_dim))
        zeros_(self.cls_token)
        self.pos_embed = self.create_parameter(shape=(1, num_patches + 1, embed_dim))  # fixed sin-cos embedding
        self.pos_embed.stop_gradient = True
        self.blocks = nn.LayerList([
            Block(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=norm_layer) for i in range(depth)
        ])
        self.norm_low = norm_layer(embed_dim)
        self.norm0 = norm_layer(embed_dim)
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        self.norm3 = norm_layer(embed_dim)
        self.norm4 = norm_layer(embed_dim)
        self.norm = norm_layer(embed_dim)         
        # --------------------------------------------------------------------------
        # Decoder specifics
        self.decoder_embed0 = nn.Linear(embed_dim, decoder_embed_dims[0], bias_attr=True)
        self.decoder_embed0_1 = nn.Linear(embed_dim, decoder_embed_dims[0], bias_attr=True)
        self.decoder_embed1 = nn.Linear(embed_dim, decoder_embed_dims[1], bias_attr=True)
        self.decoder_embed2 = nn.Linear(embed_dim, decoder_embed_dims[2], bias_attr=True)
        self.decoder_embed3 = nn.Linear(embed_dim, decoder_embed_dims[3], bias_attr=True)
        self.decoder_embed4 = nn.Linear(embed_dim, decoder_embed_dims[4], bias_attr=True)
        self.decoder_embed5 = nn.Linear(embed_dim, decoder_embed_dims[5], bias_attr=True)

        self.mask_token0 = self.create_parameter(shape=(1, 1, decoder_embed_dims[0]))
        zeros_(self.mask_token0)
        self.mask_token1 = self.create_parameter(shape=(1, 1, decoder_embed_dims[1]))
        zeros_(self.mask_token1)
        self.mask_token2 = self.create_parameter(shape=(1, 1, decoder_embed_dims[2]))
        zeros_(self.mask_token2)
        self.mask_token3 = self.create_parameter(shape=(1, 1, decoder_embed_dims[3]))
        zeros_(self.mask_token3)
        self.mask_token4 = self.create_parameter(shape=(1, 1, decoder_embed_dims[4]))
        zeros_(self.mask_token4)
        self.mask_token5 = self.create_parameter(shape=(1, 1, decoder_embed_dims[5]))
        zeros_(self.mask_token5)

        self.decoder_pos_embed0 = self.create_parameter(shape=(1, num_patches + 1, decoder_embed_dims[0]))  # fixed sin-cos embedding
        self.decoder_pos_embed0.stop_gradient = True
        self.decoder_pos_embed1 = self.create_parameter(shape=(1, num_patches + 1, decoder_embed_dims[1]))  # fixed sin-cos embedding
        self.decoder_pos_embed1.stop_gradient = True
        self.decoder_pos_embed2 = self.create_parameter(shape=(1, num_patches + 1, decoder_embed_dims[2]))  # fixed sin-cos embedding
        self.decoder_pos_embed2.stop_gradient = True
        self.decoder_pos_embed3 = self.create_parameter(shape=(1, num_patches + 1, decoder_embed_dims[3]))  # fixed sin-cos embedding
        self.decoder_pos_embed3.stop_gradient = True
        self.decoder_pos_embed4 = self.create_parameter(shape=(1, num_patches + 1, decoder_embed_dims[4]))  # fixed sin-cos embedding
        self.decoder_pos_embed4.stop_gradient = True
        self.decoder_pos_embed5 = self.create_parameter(shape=(1, num_patches + 1, decoder_embed_dims[5]))  # fixed sin-cos embedding
        self.decoder_pos_embed5.stop_gradient = True
        
        self.decoder_blocks0 = nn.LayerList([CrossBlock(
                decoder_embed_dims[0],
                decoder_num_heads,
                mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=norm_layer) for _ in range(decoder_depth)])

        self.decoder_norm0 = norm_layer(decoder_embed_dims[0])
        self.decoder_pred0 = nn.Linear(decoder_embed_dims[0], patch_size**2 * in_chans, bias_attr=True)  # decoder to patch
        if weight_norm_pred:
            self.decoder_pred0 = nn.utils.weight_norm(self.decoder_pred0)
            self.decoder_pred0.weight_g.stop_gradient = True
        
        self.decoder_blocks1 = nn.LayerList([CrossBlock(
                decoder_embed_dims[1],
                decoder_num_heads,
                mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=norm_layer) for _ in range(decoder_depth)])

        self.decoder_blocks2 = nn.LayerList([CrossBlock(
                decoder_embed_dims[2],
                decoder_num_heads,
                mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=norm_layer) for _ in range(decoder_depth)])

        self.decoder_blocks3 = nn.LayerList([CrossBlock(
                decoder_embed_dims[3],
                decoder_num_heads,
                mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=norm_layer) for _ in range(decoder_depth)])

        self.decoder_blocks4 = nn.LayerList([CrossBlock(
                decoder_embed_dims[4],
                decoder_num_heads,
                mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=norm_layer) for _ in range(decoder_depth)])

        self.decoder_blocks5 = nn.LayerList([CrossBlock(
                decoder_embed_dims[5],
                decoder_num_heads,
                mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=norm_layer) for _ in range(decoder_depth)])

        self.decoder_norm1 = norm_layer(decoder_embed_dims[1])
        self.decoder_pred1 = nn.Linear(decoder_embed_dims[1], patch_size**2 * in_chans, bias_attr=True)  # decoder to patch
        if weight_norm_pred:
            self.decoder_pred1 = nn.utils.weight_norm(self.decoder_pred1)
            self.decoder_pred1.weight_g.stop_gradient = True

        self.decoder_norm2 = norm_layer(decoder_embed_dims[2])
        self.decoder_pred2 = nn.Linear(decoder_embed_dims[2], patch_size**2 * in_chans, bias_attr=True)  # decoder to patch
        if weight_norm_pred:
            self.decoder_pred2 = nn.utils.weight_norm(self.decoder_pred2)
            self.decoder_pred2.weight_g.stop_gradient = True

        self.decoder_norm3 = norm_layer(decoder_embed_dims[3])
        self.decoder_pred3 = nn.Linear(decoder_embed_dims[3], patch_size**2 * in_chans, bias_attr=True)  # decoder to patch
        if weight_norm_pred:
            self.decoder_pred3 = nn.utils.weight_norm(self.decoder_pred3)
            self.decoder_pred3.weight_g.stop_gradient = True

        self.decoder_norm4 = norm_layer(decoder_embed_dims[4])
        self.decoder_pred4 = nn.Linear(decoder_embed_dims[4], patch_size**2 * in_chans, bias_attr=True)  # decoder to patch
        if weight_norm_pred:
            self.decoder_pred4 = nn.utils.weight_norm(self.decoder_pred4)
            self.decoder_pred4.weight_g.stop_gradient = True

        self.decoder_norm5 = norm_layer(decoder_embed_dims[5])
        self.decoder_pred5 = nn.Linear(decoder_embed_dims[5], patch_size**2 * in_chans, bias_attr=True)  # decoder to patch
        if weight_norm_pred:
            self.decoder_pred5 = nn.utils.weight_norm(self.decoder_pred5)
            self.decoder_pred5.weight_g.stop_gradient = True
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.copy_(paddle.to_tensor(pos_embed).astype(paddle.float32).unsqueeze(0), False)
        # self.t_pos_embed.copy_(paddle.to_tensor(pos_embed).astype(paddle.float32).unsqueeze(0), False)
        decoder_pos_embed0 = get_2d_sincos_pos_embed(self.decoder_pos_embed0.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed0.copy_(paddle.to_tensor(decoder_pos_embed0).astype(paddle.float32).unsqueeze(0), False)
        decoder_pos_embed1 = get_2d_sincos_pos_embed(self.decoder_pos_embed1.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed1.copy_(paddle.to_tensor(decoder_pos_embed1).astype(paddle.float32).unsqueeze(0), False)
        decoder_pos_embed2 = get_2d_sincos_pos_embed(self.decoder_pos_embed2.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed2.copy_(paddle.to_tensor(decoder_pos_embed2).astype(paddle.float32).unsqueeze(0), False)
        decoder_pos_embed3 = get_2d_sincos_pos_embed(self.decoder_pos_embed3.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed3.copy_(paddle.to_tensor(decoder_pos_embed3).astype(paddle.float32).unsqueeze(0), False)
        decoder_pos_embed4 = get_2d_sincos_pos_embed(self.decoder_pos_embed4.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed4.copy_(paddle.to_tensor(decoder_pos_embed4).astype(paddle.float32).unsqueeze(0), False)
        decoder_pos_embed5 = get_2d_sincos_pos_embed(self.decoder_pos_embed5.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed5.copy_(paddle.to_tensor(decoder_pos_embed5).astype(paddle.float32).unsqueeze(0), False)
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.reshape([self.patch_embed.proj.weight.shape[0], -1])
        xavier_uniform_(w)
        w._share_buffer_to(self.patch_embed.proj.weight)

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        normal_(self.cls_token, std=.02)
        normal_(self.mask_token0, std=.02)
        normal_(self.mask_token1, std=.02)
        normal_(self.mask_token2, std=.02)
        normal_(self.mask_token3, std=.02)
        normal_(self.mask_token4, std=.02)
        normal_(self.mask_token5, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        if self.weight_norm_pred:
            self.decoder_pred0.weight_g.fill_(1)
            self.decoder_pred1.weight_g.fill_(1)
            self.decoder_pred2.weight_g.fill_(1)
            self.decoder_pred3.weight_g.fill_(1)
            self.decoder_pred4.weight_g.fill_(1)
            self.decoder_pred5.weight_g.fill_(1)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            constant_(m.bias, 0)
            constant_(m.weight, 1.0)
            
            
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = paddle.rand([N, L])  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = paddle.argsort(noise, axis=1)  # ascend: small is keep, large is remove
        ids_restore = paddle.argsort(ids_shuffle, axis=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_dump = ids_shuffle[:, len_keep:]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = paddle.ones([N, L])
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = mask[paddle.arange(N).unsqueeze(1), ids_restore]

        return ids_keep, ids_dump, mask, ids_restore

    def forward_encoder(self, imgs, mask_ratio):
        # embed patches
        # with paddle.no_grad():
        x = self.patch_embed(imgs)
        N, L, D = x.shape  # batch, length, dim

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        ids_keep, ids_dump, mask, ids_restore = self.random_masking(x, mask_ratio)
        x = x[paddle.arange(N).unsqueeze(1), ids_keep] # [N, L*r, D] input to student 

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand([x.shape[0], -1, -1])
        x = paddle.concat((cls_tokens, x), axis=1) # [N, L*r+1, D]
        low_feat = x
        # low_feat = x.detach()
        low_feat = self.norm_low(low_feat)

        # apply Transformer blocks
        feats = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx in self.out_indices:
                feats.append(x)
        feats[0] = self.norm0(feats[0])
        feats[1] = self.norm1(feats[1])
        feats[2] = self.norm2(feats[2])
        feats[3] = self.norm3(feats[3])
        feats[4] = self.norm4(feats[4])
        latent = self.norm(x) # [N, L*r +1, D]


        return low_feat, feats, latent, mask, ids_restore, ids_dump

    def forward_decoder(self, low_feat, feats, latent, ids_restore, ids_dump):
        N = low_feat.shape[0]
        low_feat = self.decoder_embed0_1(low_feat) #[N, L*r+1, D]
        # embed tokens
        x0 = feats[0]
        x0 = self.decoder_embed0(x0)
        x0_next = x0 # used in cross att in decoder_blocks
        # append mask tokens to sequence
        mask_tokens0 = self.mask_token0.tile((x0.shape[0], ids_restore.shape[1] + 1 - x0.shape[1], 1))
        x0_ = paddle.concat([x0[:, 1:, :], mask_tokens0], axis=1)  # no cls token
        x0_ = x0_[paddle.arange(x0_.shape[0]).unsqueeze(1), ids_restore]  # unshuffle
        x0 = paddle.concat([x0[:, :1, :], x0_], axis=1)  # append cls token [N, L+1, D]
        # add pos embed
        x0 = x0 + self.decoder_pos_embed0
        # apply Transformer blocks
        for blk in self.decoder_blocks0:
            x0 = blk(x0, low_feat)
        x0 = self.decoder_norm0(x0)
        # predictor projection
        x0 = self.decoder_pred0(x0)
        # remove cls token
        x0 = x0[:, 1:, :] # [N, L, 16x16x3]
        #-------------------------------------------------------------------------
        # embed tokens
        x1 = feats[1]
        x1 = self.decoder_embed1(x1)
        x1_next = x1 # used in cross att in decoder_blocks
        # append mask tokens to sequence
        mask_tokens1 = self.mask_token1.tile((x1.shape[0], ids_restore.shape[1] + 1 - x1.shape[1], 1))
        x1_ = paddle.concat([x1[:, 1:, :], mask_tokens1], axis=1)  # no cls token
        x1_ = x1_[paddle.arange(x1_.shape[0]).unsqueeze(1), ids_restore]  # unshuffle
        x1 = paddle.concat([x1[:, :1, :], x1_], axis=1)  # append cls token
        # add pos embed
        x1 = x1 + self.decoder_pos_embed1
        # apply Transformer blocks
        kv = paddle.concat([x0_next, low_feat], axis=1)
        for blk in self.decoder_blocks1:
            x1 = blk(x1, kv)
        x1 = self.decoder_norm1(x1)
        # predictor projection
        x1 = self.decoder_pred1(x1)
        # remove cls token
        x1 = x1[:, 1:, :] # [N, L, 16x16x3]
        #-------------------------------------------------------------------------
        # embed tokens
        x2 = feats[2]
        x2 = self.decoder_embed2(x2)
        x2_next = x2 # used in cross att in decoder_blocks
        # append mask tokens to sequence
        mask_tokens2 = self.mask_token2.tile((x2.shape[0], ids_restore.shape[1] + 1 - x2.shape[1], 1))
        x2_ = paddle.concat([x2[:, 1:, :], mask_tokens2], axis=1)  # no cls token
        x2_ = x2_[paddle.arange(x2_.shape[0]).unsqueeze(1), ids_restore]  # unshuffle
        x2 = paddle.concat([x2[:, :1, :], x2_], axis=1)  # append cls token
        # add pos embed
        x2 = x2 + self.decoder_pos_embed2
        # apply Transformer blocks
        kv = paddle.concat([x1_next, x0_next, low_feat], axis=1)
        for blk in self.decoder_blocks2:
            x2 = blk(x2, kv)
        x2 = self.decoder_norm2(x2)
        # predictor projection
        x2 = self.decoder_pred2(x2)
        # remove cls token
        x2 = x2[:, 1:, :] # [N, L, 16x16x3]
        #-------------------------------------------------------------------------
        # embed tokens
        x3 = feats[3]
        x3 = self.decoder_embed3(x3)
        x3_next = x3 # used in cross att in decoder_blocks
        # append mask tokens to sequence
        mask_tokens3 = self.mask_token3.tile((x3.shape[0], ids_restore.shape[1] + 1 - x3.shape[1], 1))
        x3_ = paddle.concat([x3[:, 1:, :], mask_tokens3], axis=1)  # no cls token
        x3_ = x3_[paddle.arange(x3_.shape[0]).unsqueeze(1), ids_restore]  # unshuffle
        x3 = paddle.concat([x3[:, :1, :], x3_], axis=1)  # append cls token
        # add pos embed
        x3 = x3 + self.decoder_pos_embed3
        # apply Transformer blocks
        kv = paddle.concat([x2_next, x1_next, x0_next, low_feat], axis=1)
        for blk in self.decoder_blocks3:
            x3 = blk(x3, kv)
        x3 = self.decoder_norm3(x3)
        # predictor projection
        x3 = self.decoder_pred3(x3)
        # remove cls token
        x3 = x3[:, 1:, :] # [N, L, 16x16x3]
        #-------------------------------------------------------------------------
        # embed tokens
        x4 = feats[4]
        x4 = self.decoder_embed4(x4)
        x4_next = x4 # used in cross att in decoder_blocks
        # append mask tokens to sequence
        mask_tokens4 = self.mask_token4.tile((x4.shape[0], ids_restore.shape[1] + 1 - x4.shape[1], 1))
        x4_ = paddle.concat([x4[:, 1:, :], mask_tokens4], axis=1)  # no cls token
        x4_ = x4_[paddle.arange(x4_.shape[0]).unsqueeze(1), ids_restore]  # unshuffle
        x4 = paddle.concat([x4[:, :1, :], x4_], axis=1)  # append cls token
        # add pos embed
        x4 = x4 + self.decoder_pos_embed4
        # apply Transformer blocks
        kv = paddle.concat([x3_next, x2_next, x1_next, x0_next, low_feat], axis=1)
        for blk in self.decoder_blocks4:
            x4 = blk(x4, kv)
        x4 = self.decoder_norm4(x4)
        # predictor projection
        x4 = self.decoder_pred4(x4)
        # remove cls token
        x4 = x4[:, 1:, :] # [N, L, 16x16x3]
        #--------------------------------------------------------------------------
        # embed tokens
        x5 = latent
        x5 = self.decoder_embed5(x5)
        # append mask tokens to sequence
        mask_tokens5 = self.mask_token5.tile((x5.shape[0], ids_restore.shape[1] + 1 - x5.shape[1], 1))
        x5_ = paddle.concat([x5[:, 1:, :], mask_tokens5], axis=1)  # no cls token
        x5_ = x5_[paddle.arange(x5_.shape[0]).unsqueeze(1), ids_restore]  # unshuffle
        x5 = paddle.concat([x5[:, :1, :], x5_], axis=1)  # append cls token
        # add pos embed
        x5 = x5 + self.decoder_pos_embed5
        # apply Transformer blocks
        kv = paddle.concat([x4_next, x3_next, x2_next, x1_next, x0_next, low_feat], axis=1)
        for blk in self.decoder_blocks5:
            x5 = blk(x5, kv)
        x5 = self.decoder_norm5(x5) # [N, L+1, D]
        # predictor projection
        x5 = self.decoder_pred5(x5)
        # remove cls token
        x5 = x5[:, 1:, :] # [N, L, 16x16x3]


        return x0, x1, x2, x3, x4, x5

    def forward_loss(self, imgs, preds, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        N = imgs.shape[0]
        target = patchify(imgs, self.patch_embed.patch_size[0])
        # if self.norm_pix_loss:
        mean = target.mean(axis=-1, keepdim=True)
        var = target.var(axis=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5
        
        reconstr_loss0 = (preds[0] + preds[-1] - target)**2
        reconstr_loss0 = reconstr_loss0.mean(axis=-1)  # [N, L], mean loss per patch
        reconstr_loss0 = (reconstr_loss0 * mask).sum() / mask.sum()  # mean loss on removed patches

        reconstr_loss1 = (preds[1] + preds[-2] - target)**2
        reconstr_loss1 = reconstr_loss1.mean(axis=-1)  # [N, L], mean loss per patch
        reconstr_loss1 = (reconstr_loss1 * mask).sum() / mask.sum()  # mean loss on removed patches

        reconstr_loss2 = (preds[2] + preds[-3] - target)**2
        reconstr_loss2 = reconstr_loss2.mean(axis=-1)  # [N, L], mean loss per patch
        reconstr_loss2 = (reconstr_loss2 * mask).sum() / mask.sum()  # mean loss on removed patches

        loss = (reconstr_loss0 + reconstr_loss1 + reconstr_loss2) / 3.

        # # Visulization------------------------------------------------------------
        # mask_ = mask_unpatchify(mask, 14, 14, 16)
        
        # pred_img1 = (preds[0] + preds[-1]) * (var + 1.e-6)**.5 + mean
        # pred_img2 = (preds[1] + preds[-2]) * (var + 1.e-6)**.5 + mean
        # pred_img3 = (preds[2] + preds[-3]) * (var + 1.e-6)**.5 + mean

        # main_img1 = preds[0] * (var + 1.e-6)**.5 + mean
        # res_img1 = preds[-1] * (var + 1.e-6)**.5 + mean

        # main_img2 = preds[1] * (var + 1.e-6)**.5 + mean
        # res_img2 = preds[-2] * (var + 1.e-6)**.5 + mean

        # main_img3 = preds[2] * (var + 1.e-6)**.5 + mean
        # res_img3 = preds[-3] * (var + 1.e-6)**.5 + mean

 
        # return loss, [reconstr_loss0, reconstr_loss1, reconstr_loss2], [pred_img1, pred_img2, pred_img3,
        #  main_img1, res_img1, main_img2, res_img2, main_img3, res_img3]
        return loss, [reconstr_loss0, reconstr_loss1, reconstr_loss2], [None]

    def forward(self, imgs, mask_ratio=0.75):
        low_feat, feats, latent, mask, ids_restore, ids_dump = self.forward_encoder(imgs, mask_ratio)
        preds = self.forward_decoder(low_feat, feats, latent, ids_restore, ids_dump) # [N, L, p*p*3]
        loss, loss_terms, pred_imgs = self.forward_loss(imgs, preds, mask)

        
        return loss, loss_terms, pred_imgs, mask


def mirl_vit_small_patch16_dec384d2b(**kwargs):
    model = MIRLViT(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        decoder_embed_dims=[384, 384, 384],
        decoder_depth=2,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        out_indices=[3, 8],
        weight_norm_pred=True,
        **kwargs)
    return model

def mirl_vit_small_patch16_depth54_dec384d1b(**kwargs):
    model = MIRLViT(
        patch_size=16,
        embed_dim=384,
        depth=54,
        num_heads=12,
        decoder_embed_dims=[384, 384, 384, 384, 384, 384],
        decoder_depth=1,
        decoder_num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        out_indices=[8, 17, 26, 35, 44],
        weight_norm_pred=True,
        **kwargs)
    return model


def mirl_vit_base_patch16_depth48_dec512d2b(**kwargs):
    model = MIRLViT(
        patch_size=16,
        embed_dim=768,
        depth=48,
        num_heads=12,
        decoder_embed_dims=[512, 512, 512, 512, 512, 512],
        decoder_depth=2,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        out_indices=[7, 15, 23, 31, 39],
        weight_norm_pred=True,
        **kwargs)
    return model

def mirl_vit_base_patch16_dec512d2b(**kwargs):
    model = MIRLViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dims=[512, 512, 512, 512],
        decoder_depth=2,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        out_indices=[5, 11, 17, 23],
        weight_norm_pred=True,
        **kwargs)
    return model



# set recommended archs
mirl_vit_small_patch16 = mirl_vit_small_patch16_dec384d2b # decoder: 384 dim, 2 blocks
mirl_vit_small_patch16_depth54 = mirl_vit_small_patch16_depth54_dec384d1b # decoder: 384 dim, 1 blocks
mirl_vit_base_patch16 = mirl_vit_base_patch16_dec512d2b  # decoder: 512 dim, 2 blocks
mirl_vit_base_depth48_patch16 = mirl_vit_base_patch16_depth48_dec512d2b  # decoder: 512 dim, 2 blocks


