import math
import os
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from torchvision.utils import save_image
import random
import time


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,
                    epoch_id=0,
                    wandb=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    loss_value=0
    for data_iter_step, (imgs,masks) in enumerate(metric_logger.log_every(data_loader, print_freq, header,wandb)):
        imgs=imgs.squeeze(0)
        masks=masks.squeeze(0)
        imgs = imgs.to(device)
        masks = masks.to(device)
        t0 = time.time()
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        #print(imgs.shape,masks.shape)
        with torch.cuda.amp.autocast():
            # loss, loss_mse, loss_lpips = model.loss(samples,require_lpips=True)
            if not args.mask_loss:
                loss = model.loss(imgs)
            else:
                if args.distributed:
                    loss_mse,loss_mask=model.module.loss(imgs,mask=masks,epoch_id=epoch_id)
                else:
                    loss_mse, loss_mask = model.loss(imgs, mask=masks, epoch_id=epoch_id)
                loss=loss_mse+loss_mask
                metric_logger.update(loss_mse=loss_mse.item())
                metric_logger.update(loss_mask=loss_mask.item())

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
        t2 = time.time()
        if args.report_time:
            print(t1 - t0, t2 - t1)
        # loss.backward()
        # optimizer.step()
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        # metric_logger.update(loss_mse=loss.detach().item())
        # metric_logger.update(loss_lpips=loss.detach().item())

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
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
    # switch to evaluation mode
    model.eval()
    # compute output
    with torch.cuda.amp.autocast():
        output, gt = model(images,masks,num=128)
    os.makedirs(args.log_dir + '/imgs', exist_ok=True)
    save_image(torch.cat([output[:8, :3, :, :], gt[:8]], dim=0), args.log_dir + '/imgs/%d.jpg' % epoch_id, nrow=8,
               normalize=False)
    # if wandb is not None:
    #     wandb.log({"image": wandb.Image(args.log_dir + '/imgs/%d.jpg' % epoch_id)})
    # if model.control_num:
    #     for num in [16,32, 64]:
    #         with torch.cuda.amp.autocast():
    #             output, gt = model(images, num)
    #         print(output.shape, gt.shape)
    #         os.makedirs(args.log_dir + '/imgs', exist_ok=True)
    #         save_image(torch.cat([output[:8, :3, :, :], gt[:8]], dim=0),
    #                    args.log_dir + '/imgs/%d-%d.jpg' % (epoch_id, num), nrow=8,
    #                    normalize=False)