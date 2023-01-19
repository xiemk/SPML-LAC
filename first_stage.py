import argparse
import math
import os, sys
import random
import time
import json
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.optim
import torch.utils.data

from torch.utils.tensorboard import SummaryWriter

import _init_paths
from dataset.get_dataset import get_datasets

from models.LEModel import build_LEModel

from utils.logger import setup_logger
from utils.meter import AverageMeter, AverageMeterHMS, ProgressMeter
from utils.helper import clean_state_dict, function_mAP, get_raw_dict, ModelEma


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def parser_args():
    parser = argparse.ArgumentParser(description='First Training')
    

    # data
    parser.add_argument('--dataset_name', help='dataset name', default='coco', choices=['voc', 'coco', 'nus', 'cub'])
    parser.add_argument('--dataset_dir', help='dir of all datasets', default='./data')
    parser.add_argument('--img_size', default=448, type=int,
                        help='size of input images')
    parser.add_argument('--output', metavar='DIR', default='./outputs',
                        help='path to output folder')


    # loss
    parser.add_argument('--lambda_plc', default=1, type=float,
                        help='coefficient of PLC loss')
    parser.add_argument('--threshold', default=0.6, type=float,
                        help='threshold')


    # train
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--val_interval', default=1, type=int, metavar='N',
                        help='interval of validation')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='batch size')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight_decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print_freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='apply amp')
    parser.add_argument('--early_stop', action='store_true', default=True,
                        help='apply early stop')


    # random seed
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')


    # model
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=True,
                        help='use pre-trained model. default is True. ')
    parser.add_argument('--feat_dim', default=128, type=int,
                        help="Size of the low-dimensional embeddings")
    parser.add_argument('--is_proj', action='store_true', default=False,
                        help='on/off projection')
    parser.add_argument('--is_data_parallel', action='store_true', default=False,
                        help='on/off nn.DataParallel()')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume_omit', default=[], type=str, nargs='*')
    parser.add_argument('--ema_decay', default=0.9997, type=float, metavar='M',
                        help='decay of model ema')


    args = parser.parse_args()

    args.dataset_dir = os.path.join(args.dataset_dir, args.dataset_name) 
    args.output = os.path.join(args.output, args.dataset_name, 'first')

    return args


def get_args():
    args = parser_args()
    return args


def main():
    args = get_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    os.makedirs(args.output, exist_ok=True)

    logger = setup_logger(output=args.output, color=False, name="LEModel")
    logger.info("Command: "+' '.join(sys.argv))

    path = os.path.join(args.output, "config.json")
    with open(path, 'w') as f:
        json.dump(get_raw_dict(args), f, indent=2)
    logger.info("Full config saved to {}".format(path))

    return main_worker(args, logger)

def main_worker(args, logger):

    # build model
    model = build_LEModel(args)
    if args.is_data_parallel:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            if 'state_dict' in checkpoint:
                state_dict = clean_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                state_dict = clean_state_dict(checkpoint['model'])
            else:
                raise ValueError("No model or state_dicr Found!!!")
            logger.info("Omitting {}".format(args.resume_omit))
            for omit_name in args.resume_omit:
                del state_dict[omit_name]
            model.load_state_dict(state_dict, strict=False)
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            del state_dict
            torch.cuda.empty_cache() 
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    ema_m = ModelEma(model, args.ema_decay) # 0.9997

    # optimizer
    args.lr_mult = args.batch_size / 256

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
    ]
    optimizer = getattr(torch.optim, 'AdamW')(
        param_dicts,
        args.lr_mult * args.lr,
        betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
    )

    # tensorboard
    summary_writer = SummaryWriter(log_dir=args.output)


    # Data loading code
    train_dataset, val_dataset, test_dataset = get_datasets(args)
    print("len(train_dataset):", len(train_dataset)) 
    print("len(val_dataset):", len(val_dataset))
    print("len(test_dataset):", len(test_dataset))

    args.workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)


    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    
    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    mAPs = AverageMeter('mAP', ':5.5f', val_only=True)
    mAPs_ema = AverageMeter('mAP_ema', ':5.5f', val_only=True)
    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, mAPs, mAPs_ema],
        prefix='=> Test Epoch: ')

    # one cycle learning rate
    args.steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, pct_start=0.2)


    end = time.time()
    best_epoch = -1
    best_regular_mAP = 0
    best_regular_epoch = -1
    best_ema_mAP = 0
    regular_mAP_list = []
    ema_mAP_list = []
    mAP_ema_test = 0
    best_mAP = 0


    torch.cuda.empty_cache()
    for epoch in range(args.start_epoch, args.epochs):

        torch.cuda.empty_cache()

        # train for one epoch
        loss = train(train_loader, model, ema_m, optimizer, scheduler, epoch, args, logger)

        if summary_writer:
            # tensorboard logger
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % args.val_interval == 0:

            # evaluate on validation set
            mAP = validate(val_loader, model, args, logger)
            mAP_ema = validate(val_loader, ema_m.module, args, logger)
            mAPs.update(mAP)
            mAPs_ema.update(mAP_ema)
            epoch_time.update(time.time() - end)
            end = time.time()
            eta.update(epoch_time.avg * (args.epochs - epoch - 1))

            regular_mAP_list.append(mAP)
            ema_mAP_list.append(mAP_ema)

            progress.display(epoch, logger)

            if summary_writer:
                # tensorboard logger
                summary_writer.add_scalar('val_mAP', mAP, epoch)
                summary_writer.add_scalar('val_mAP_ema', mAP_ema, epoch)

            # remember best (regular) mAP and corresponding epochs
            if mAP > best_regular_mAP:
                best_regular_mAP = max(best_regular_mAP, mAP)
                best_regular_epoch = epoch
            if mAP_ema > best_ema_mAP:
                best_ema_mAP = max(mAP_ema, best_ema_mAP)
            
            if mAP_ema > mAP:
                mAP = mAP_ema
                state_dict = ema_m.module.state_dict()
            else:
                state_dict = model.state_dict()
            is_best = mAP > best_mAP
            if is_best:
                best_epoch = epoch
            best_mAP = max(mAP, best_mAP)

            if best_mAP == mAP_ema:
                mAP_ema_test = validate(test_loader, ema_m.module, args, logger)
            elif best_mAP == mAP:
                mAP_ema_test = validate(test_loader, model, args, logger)

            logger.info("{} | Set best mAP {} in ep {}".format(epoch, best_mAP, best_epoch))
            logger.info("   | best regular mAP {} in ep {}".format(best_regular_mAP, best_regular_epoch))
            logger.info("   | best test mAP {} ".format(mAP_ema_test))

           
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'best_mAP': best_mAP,
                'optimizer' : optimizer.state_dict(),
            }, is_best=is_best, filename=os.path.join(args.output, 'checkpoint.pth.tar'))

            if math.isnan(loss):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_mAP': best_mAP,
                    'optimizer' : optimizer.state_dict(),
                }, is_best=is_best, filename=os.path.join(args.output, 'checkpoint_nan.pth.tar'))
                logger.info('Loss is NaN, break')
                sys.exit(1)


            # early stop
            if args.early_stop:
                if best_epoch >= 0 and epoch - max(best_epoch, best_regular_epoch) > 8:
                    if len(ema_mAP_list) > 1 and ema_mAP_list[-1] < best_ema_mAP:
                        logger.info("epoch - best_epoch = {}, stop!".format(epoch - best_epoch))
                        break

    print("Best mAP:", best_mAP)

    if summary_writer:
        summary_writer.close()
    
    return 0

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if is_best:
        torch.save(state, os.path.split(filename)[0] + '/model_best.pth.tar')

def train(train_loader, model, ema_m, optimizer, scheduler, epoch, args, logger):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    loss_an = AverageMeter('L_an', ':5.3f')
    loss_plc = AverageMeter('L_plc', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        args.steps_per_epoch,
        [loss_an, loss_plc, lr, losses, mem],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    # switch to train mode
    model.train()

    for i, ((inputs_w, inputs_s), targets) in enumerate(train_loader):

        # **********************************************compute loss*************************************************************

        batch_size = inputs_w.size(0)

        inputs = torch.cat([inputs_w, inputs_s], dim=0).cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True).float()
        # mixed precision ---- compute outputs
        with torch.cuda.amp.autocast(enabled=args.amp):
            logits = model(inputs)
         
        logits_w, logits_s = torch.split(logits[:], batch_size)

        L_an = F.binary_cross_entropy_with_logits(logits_w, targets, reduction='mean')

        pseudo_label = torch.sigmoid(logits_w.detach()) + targets
        pseudo_label_mask = ((pseudo_label >= args.threshold) | (pseudo_label < (1 - args.threshold))).float()
        pseudo_label_hard = (pseudo_label >= args.threshold).float()


        L_plc = (F.binary_cross_entropy_with_logits(logits_s, pseudo_label_hard, reduction='none') * pseudo_label_mask).sum() / pseudo_label_mask.sum()

        loss = L_an + args.lambda_plc * L_plc

        # *********************************************************************************************************************

        # record loss
        loss_an.update(L_an.item(), inputs.size(0))
        loss_plc.update(L_plc.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # one cycle learning rate
        scheduler.step()
        lr.update(get_learning_rate(optimizer))
        ema_m.update(model)


        if i % args.print_freq == 0:
            progress.display(i, logger)

    return losses.avg


@torch.no_grad()
def validate(val_loader, model, args, logger):
    batch_time = AverageMeter('Time', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, mem],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    outputs_sm_list = []
    targets_list = []
        
    end = time.time()
    for i, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs_sm = torch.sigmoid(model(inputs))
        
        # add list
        outputs_sm_list.append(outputs_sm.detach().cpu())
        targets_list.append(targets.detach().cpu())

        # record memory
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, logger)

    # calculate mAP
    mAP = function_mAP(torch.cat(targets_list).numpy(), torch.cat(outputs_sm_list).numpy())
    
    print("Calculating mAP:")  
    logger.info("  mAP: {}".format(mAP))

    return mAP


if __name__ == '__main__':
    main()