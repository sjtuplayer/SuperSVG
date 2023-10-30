import math
import sys
from typing import Iterable

import torch
import util.misc as misc
import util.lr_sched as lr_sched
import time
import cv2
import numpy as np

def get_mask_img(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img = torch.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 1))
    # img[:,:,1] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = torch.tensor([1.])
        img[m] = color_mask
    return img

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
      
        samples = samples.to(device, non_blocking=True)
        density_list = []
        for idx in range(samples.shape[0]):
            input_GRAY = cv2.cvtColor(samples[idx].detach().cpu().permute(1,2,0).numpy(), cv2.COLOR_BGR2GRAY)
            gradient_x = cv2.Sobel(input_GRAY, cv2.CV_32FC1, 1, 0, ksize=5)
            gradient_y = cv2.Sobel(input_GRAY, cv2.CV_32FC1, 0, 1, ksize=5)
            gradient_magnitude = np.sqrt(gradient_x ** 2.0 + gradient_y ** 2.0)
            density = cv2.blur(gradient_magnitude, (14, 14))
            density = cv2.normalize(density, None, 0.01, 1.0, cv2.NORM_MINMAX) # 归一化
            density_list.append(torch.tensor(density).unsqueeze(0))

        masks = torch.stack(density_list, dim=0).to(device)
        with torch.cuda.amp.autocast():
            loss, _, loss_l2, loss_density, loss_lpips = model.density_loss(samples, masks)
       
        loss_value = loss.item()
        loss_l2_value = loss_l2.item()
        loss_density_value = loss_density.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.encoder.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_l2_value_reduce = misc.all_reduce_mean(loss_l2_value)
        loss_density_value_reduce = misc.all_reduce_mean(loss_density_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss_l2', loss_l2_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss_density', loss_density_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    evaluate(model, samples, masks, log_writer)

    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, images, masks, log_writer):
    model.eval()
    with torch.cuda.amp.autocast():
        output, gt = model(images, masks)

    batch_size = images.shape[0]
    for i in range(batch_size):
        log_writer.add_image('image_' + str(i) + '_pred', output[i])
        log_writer.add_image('image_' + str(i) + '_gt', gt[i])
        log_writer.add_image('image_' + str(i) + '_mask', masks[i])