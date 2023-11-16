import math
import sys
from typing import Iterable

import paddle

import util.misc as misc
import util.lr_sched as lr_sched
from util.functions import adjust_ema_momentum, patchify, unpatchify, show_image, mask_unpatchify, plot_vis


def train_one_epoch(model,
                    data_loader,
                    optimizer,
                    device,
                    epoch,
                    loss_scaler,
                    log_writer=None,
                    args=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('total_loss', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('reconstr_loss0', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('reconstr_loss1', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('reconstr_loss2', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.clear_grad()
    ### step ema
    curr_ema = adjust_ema_momentum(epoch, args)
    print ('Dynamic EMA DECAY ', curr_ema)

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        with paddle.amp.auto_cast():
            loss, loss_terms, pred_imgs, mask= model(samples, mask_ratio=args.mask_ratio, teacher_ema=curr_ema)
        loss_value = loss.item()
        if (log_writer is not None) and (data_iter_step % 100 == 0) and (pred_imgs[0] is not None):
            plot_vis(samples, pred_imgs, mask, save_dir=f'{args.output_dir}/images',  name=f'{int(epoch * len(data_loader) + data_iter_step)}')
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        total_grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=None,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (log_writer is not None) and ((data_iter_step + 1) % (print_freq * 10) == 0):
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            for tag, value in model.named_parameters():
                if (value is not None) and (value.grad is not None):
                    tag = tag.replace('.', '/')
                    try:
                        log_writer.add_scalar('grads/'+tag, value.grad.detach().abs().mean().numpy(), epoch_1000x)
                    except:
                        print(f'{tag} is NaN or Inf')
            try:
                log_writer.add_scalar('grads/total_grad_norm', total_grad_norm.detach().numpy(), epoch_1000x)
            except:
                print('total_grad_norm is NaN or Inf')
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.clear_grad()

        paddle.device.cuda.synchronize()

        metric_logger.update(total_loss=loss_value)
        metric_logger.update(reconstr_loss0=loss_terms[0].item())
        metric_logger.update(reconstr_loss1=loss_terms[1].item())
        metric_logger.update(reconstr_loss2=loss_terms[1].item())


        lr = optimizer._param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if (log_writer is not None) and ((data_iter_step + 1) % print_freq == 0):
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('step/train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('step/lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
                      
