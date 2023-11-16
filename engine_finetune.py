import math
import sys
from typing import Iterable, Optional

import paddle

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model,
                    criterion,
                    data_loader,
                    optimizer,
                    device,
                    epoch,
                    loss_scaler,
                    max_norm=0,
                    log_writer=None,
                    model_ema=None,
                    args=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', misc.SmoothedValue(
            window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'loss', misc.SmoothedValue(
            window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.clear_grad()

    # if log_writer is not None:
    #     print('log_dir: {}'.format(log_writer.kwargs['log_dir']))

    for data_iter_step, (
            samples, targets
    ) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args)

        with paddle.amp.auto_cast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        total_grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0 and (data_iter_step + 1) % print_freq == 0:
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
            if model_ema is not None:
                model_ema.update(model)
            optimizer.clear_grad()

        paddle.device.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('step/loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('step/lr', max_lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@paddle.no_grad()
def evaluate(data_loader, model, device):
    criterion = paddle.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        # compute output
        with paddle.amp.auto_cast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = misc.accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(
            top1=metric_logger.acc1,
            top5=metric_logger.acc5,
            losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
 
