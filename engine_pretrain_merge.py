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
width=128
block=2
# class memory:
#     def __init__(self):
#         self.states=[]
#         self.l=5000
#     def append(self,state,step=None,alpha_canvas=None,device='cpu'):
#         if device=='cpu':
#             state=state.cpu()
#         if alpha_canvas is None:
#             self.states.append((state.detach(),step))
#         else:
#             self.states.append((state.detach(), step,alpha_canvas))
#         if len(self.states)>self.l:
#             self.states.pop(random.randint(0,self.l-1))
#     def random_reset(self):
#         random.shuffle(self.states)
#     def pop(self):
#         return self.states.pop()
#     def empty(self):
#         if len(self.states)==0:
#             return True
#         else:
#             return False
#
# def prepare(img,model0,width,block):
#     img = img.resize(1, 3, block, width, block, width)
#     img = img.permute(0, 2, 4, 1, 3, 5)
#     img = img.resize(block * block, 3, width, width)


def train_one_epoch(model: torch.nn.Module,
                    model0: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,
                    epoch_id=0):
    model.train(True)
    model0.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    #block_num=block*block

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples = samples.to(device, non_blocking=True)
        batch_size=samples.size(0)
        img = samples.resize(batch_size, 3, block, width, block, width)
        img = img.permute(0, 2, 4, 1, 3, 5)
        img = img.resize(batch_size*block * block, 3, width, width)
        paths=model0.predict_path(img)
        # paths[::block_num,:-3]=paths[::block_num,:-3]/2
        # paths[1::block_num, :-3:2] = paths[1::block_num, :-3:2] / 2
        # paths[1::block_num, 1:-3:2] = 0.5+paths[1::block_num, 1:-3:2] / 2
        # paths[2::block_num, :-3:2] = 0.5+paths[2::block_num, :-3:2] / 2
        # paths[2::block_num, 1:-3:2] = paths[2::block_num, 1:-3:2] / 2
        # paths[3::block_num, :-3] = paths[3::block_num, :-3] / 2
        paths=paths.resize(batch_size,block*block*128,27)
        with torch.cuda.amp.autocast():
            #loss, loss_mse, loss_lpips = model.loss(samples,require_lpips=True)
            # loss=model.encoder.stroke_head.stroke_tokens.sum()
            # print(loss)
            loss = model.loss(paths,samples)

        loss_value = loss.item()


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss /= accum_iter
        # loss_scaler(loss, optimizer, parameters=model.encoder.parameters(),
        #             update_grad=(data_iter_step + 1) % accum_iter == 0)
        loss.backward()
        optimizer.step()

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
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    evaluate(model,model0, samples, log_writer,args,epoch_id)

    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model,model0, images, log_writer,args,epoch_id):
    # switch to evaluation mode
    model.eval()
    model0.eval()
    # compute output
    with torch.cuda.amp.autocast():
        batch_size = images.size(0)
        img = images.resize(batch_size, 3, block, width, block, width)
        img = img.permute(0, 2, 4, 1, 3, 5)
        img = img.resize(batch_size * block * block, 3, width, width)
        paths = model0.predict_path(img)
        paths = paths.resize(batch_size, block * block * 128, 27)
        output, gt = model(paths,images)
    os.makedirs(args.log_dir+'/imgs',exist_ok=True)
    save_image(torch.cat([output[:8,:3,:,:],gt[:8]],dim=0),args.log_dir+'/imgs/%d.jpg'%epoch_id,nrow=8,normalize=False)
# @torch.no_grad()
# def test_path(model,model0, images, log_writer,args,epoch_id):
#     # switch to evaluation mode
#     model.eval()
#     model0.eval()
#     # compute output
#     block_num=block*block
#     with torch.cuda.amp.autocast():
#         batch_size = images.size(0)
#         img = images.resize(batch_size, 3, block, width, block, width)
#         img = img.permute(0, 2, 4, 1, 3, 5)
#         img = img.resize(batch_size * block * block, 3, width, width)
#         paths = model0.predict_path(img)
#         paths[::block_num, :-3] = paths[::block_num, :-3] / 2
#         paths[1::block_num, :-3:2] = paths[1::block_num, :-3:2] / 2
#         paths[1::block_num, 1:-3:2] = 0.5 + paths[1::block_num, 1:-3:2] / 2
#         paths[2::block_num, :-3:2] = 0.5 + paths[2::block_num, :-3:2] / 2
#         paths[2::block_num, 1:-3:2] = paths[2::block_num, 1:-3:2] / 2
#         paths[3::block_num, :-3] = paths[3::block_num, :-3] / 2
#         paths = paths.resize(batch_size, block * block * 128, 27)
#         out=model.rendering(paths)
#     print(((out[:,:3,:,:]-img)**2).mean())
#     os.makedirs(args.log_dir+'/imgs',exist_ok=True)
#     save_image(torch.cat([out[:8,:3,:,:],img[:8]],dim=0),args.log_dir+'/imgs/%d.jpg'%epoch_id,nrow=8,normalize=False)

