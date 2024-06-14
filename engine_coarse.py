import math
import os
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from torchvision.utils import save_image
import time


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,
                    epoch_id=0,
                    wandb=None,
                    scheduler=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, (imgs,masks) in enumerate(metric_logger.log_every(data_loader, print_freq, header,wandb)):
        imgs=imgs.squeeze(0)
        masks=masks.squeeze(0)
        imgs = imgs.to(device)
        masks = masks.to(device)
        if scheduler is None:
            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        else:
            scheduler.step(data_iter_step / len(data_loader) + epoch)

        with torch.cuda.amp.autocast():
            if args.distributed:
                loss,kwargs=model.module.loss(imgs,mask=masks,epoch_id=epoch_id)
            else:
                loss,kwargs = model.loss(imgs, mask=masks, epoch_id=epoch_id)
        metric_logger.update(**kwargs)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss /= accum_iter
        t1 = time.time()
        if args.distributed:
            loss_scaler(loss, optimizer, parameters=model.module.encoder.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
        else:
            loss_scaler(loss, optimizer, parameters=model.encoder.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()


        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    evaluate(model, imgs, masks, log_writer, args, epoch_id,wandb=wandb)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)




    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, images,masks, log_writer, args, epoch_id,wandb):
    model.eval()
    with torch.cuda.amp.autocast():
        output, gt = model(images,masks,num=128)
    os.makedirs(args.log_dir + '/imgs', exist_ok=True)
    save_image(torch.cat([output[:8, :3, :, :], gt[:8]], dim=0), args.log_dir + '/imgs/%d.jpg' % epoch_id, nrow=8,
               normalize=False)