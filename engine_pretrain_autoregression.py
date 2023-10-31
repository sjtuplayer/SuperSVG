import math
import os
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from torchvision.utils import save_image
import random


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,
                    epoch_id=0,
                    critic=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            #loss, loss_mse, loss_lpips = model.loss(samples,require_lpips=True)
            loss,record_dict = model.loss_stroke(samples,critic=critic,epoch_id=epoch_id)

        loss_value = loss.item()


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss /= accum_iter

        loss_scaler(loss, optimizer, parameters=model.encoder.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        # loss.backward()
        # optimizer.step()
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        # for key in record_dict.keys():
        metric_logger.update(**record_dict)
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
    evaluate(model, samples, log_writer, args, epoch_id)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)


    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, images, log_writer,args,epoch_id):
    # switch to evaluation mode
    model.eval()
    # compute output
    os.makedirs(args.log_dir + '/imgs', exist_ok=True)
    with torch.cuda.amp.autocast():
        canvas=torch.zeros_like(images).to(images.device)
        for i in range(3):
            canvas, gt,pred = model(images,canvas,step=i)
            print(i,((canvas-gt)**2).mean())
            save_image(torch.cat([canvas[:8,:3,:,:],gt[:8],pred[:8,:3,:,:]],dim=0),args.log_dir+'/imgs/%d-%d.jpg'%(epoch_id,i),nrow=8,normalize=False)