"""functions for supporting MIM training"""

import os
import numpy as np
import matplotlib.pyplot as plt
import paddle

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):

    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def adjust_ema_momentum(epoch, args):
    if epoch < 100:
        tmp_ema_decay = args.teacher_ema + epoch / 100 * (0.9999 - args.teacher_ema)
    else:
        tmp_ema_decay = 0.9999 + min(300, epoch-100)/300 * (0.99999 - 0.9999)
    return tmp_ema_decay


def patchify(imgs, p):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    # p = self.patch_embed.patch_size[0]
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = paddle.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
    return x


def unpatchify(x, p):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    # p = self.patch_embed.patch_size[0]
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = paddle.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs
    
def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    plt.imshow(np.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).astype(np.int32))
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def mask_unpatchify(x, h, w, p):
    """
    x: (N, L, patch_size**2 *3)
    """
    
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, 1))
    x = paddle.tile(x, repeat_times=[1, 1, 1, p * p * 3])
    x = x.reshape([x.shape[0], x.shape[1], x.shape[2], p, p, 3])
    x = paddle.einsum('nhwpqc->nchpwq', x)
    x = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
    return x


def plot_vis(imgs, pred_imgs, mask, save_dir, name):
    """ Visulization
    """

    # make the plt figure larger
    # plt.rcParams['figure.figsize'] = [10, 10]
    fig = plt.figure(figsize=(10, 10))

    _mask = mask_unpatchify(mask, 14, 14, 16)
    # pred_img1, pred_img2, pred_img3, main_img1, res_img1, main_img2, res_img2, main_img3, res_img3
    pred_img1  = unpatchify(pred_imgs[0], 16).numpy()[0].transpose([1,2,0])
    pred_img2  = unpatchify(pred_imgs[1], 16).numpy()[0].transpose([1,2,0])
    pred_img3  = unpatchify(pred_imgs[2], 16).numpy()[0].transpose([1,2,0])
    main_img1  = unpatchify(pred_imgs[3], 16).numpy()[0].transpose([1,2,0])
    res_img1  = unpatchify(pred_imgs[4], 16).numpy()[0].transpose([1,2,0])
    main_img2  = unpatchify(pred_imgs[5], 16).numpy()[0].transpose([1,2,0])
    res_img2  = unpatchify(pred_imgs[6], 16).numpy()[0].transpose([1,2,0])
    main_img3  = unpatchify(pred_imgs[7], 16).numpy()[0].transpose([1,2,0])
    res_img3  = unpatchify(pred_imgs[8], 16).numpy()[0].transpose([1,2,0])




    _mask = mask_unpatchify(mask, 14, 14, 16)
    _mask = _mask.numpy()[0].transpose([1, 2, 0])
    img = imgs[0].transpose([1, 2, 0]).numpy()
    plt.subplot(4, 3, 1)
    show_image(img, "original")
    plt.subplot(4, 3, 2)
    show_image(img * (1-_mask), "masked")
    plt.subplot(4, 3, 3)
    show_image(img * (1-_mask), "masked")

    plt.subplot(4, 3, 4)
    show_image(pred_img1 , f"pred_img1")
    plt.subplot(4, 3, 5)
    show_image(main_img1 , f"main_img1")
    plt.subplot(4, 3, 6)
    show_image(res_img1 , f"res_img1")

    plt.subplot(4, 3, 7)
    show_image(pred_img2 , f"pred_img2")
    plt.subplot(4, 3, 8)
    show_image(main_img2 , f"main_img2")
    plt.subplot(4, 3, 9)
    show_image(res_img2, f"res_img2")

    plt.subplot(4, 3, 10)
    show_image(pred_img3 , f"pred_img3")
    plt.subplot(4, 3, 11)
    show_image(main_img3 , f"main_img3")
    plt.subplot(4, 3, 12)
    show_image(res_img3, f"res_img3")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f'{save_dir}/{name}.jpg', dpi=200, bbox_inches = 'tight')




