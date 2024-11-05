import os
import datetime
import random
import time
import cv2
import numpy as np
import logging
import argparse
#from visdom import Visdom
import os.path as osp
from util import imutils
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.cuda.amp import autocast as autocast
from torch.cuda import amp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from model.few_seg import MyFSNet
# from model.workdir import
from torch.nn import DataParallel
from dataset import iSAID, iSAID_1,LoveDA

from util import config
from util.util import AverageMeter,  intersectionAndUnionGPU, get_model_para_number, setup_seed, get_logger, get_save_path, \
                                     fix_bn, check_makedirs,freeze_modules,lr_decay, Special_characters

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
torch.autograd.set_detect_anomaly(True)

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Few-Shot Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='MyFSNet', help='') #
    parser.add_argument('--shot', type=int, default=1, help='') #
    parser.add_argument('--split', type=int, default=0, help='') # 
    parser.add_argument('--dataset', type=str, default='iSAID', help='') #  iSAID
    parser.add_argument('--backbone', type=str, default='vgg', help='') #
    parser.add_argument('--variable1', type=str, default='', help='') #
    parser.add_argument('--variable2', type=str, default='', help='') #
    parser.add_argument('--mode', default="train", action='store_true')
    parser.add_argument('--use_cross',type=bool,default=False)
    parser.add_argument('--use_self', type=bool,default=True)
    parser.add_argument('--local_rank', type=int, default=-1, help='number of cpu threads to use during batch generation')    
    parser.add_argument('--opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    #base_config = 'config/base.yaml'
    data_config = 'config/dataset/{}.yaml'.format(args.dataset)
    if args.arch in ['MyFSNet']:
        model_config = 'config/model/few_seg/{}.yaml'.format(args.arch)
    else:
        model_config = 'config/model/workdir/{}.yaml'.format(args.arch)

    if os.path.exists(model_config):
        cfg = config.load_cfg_from_cfg_file([data_config, model_config])
    else:
        cfg = config.load_cfg_from_cfg_file([data_config])

    cfg = config.merge_cfg_from_args(cfg, args)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_model(args,device):

    model = eval(args.arch).OneModel(args, cls_type='Base')
    #optimizer = model.get_optim(model, args.lr_decay, LR=args.base_lr)#SGD
    '''
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "transformer" not in n
                    and p.requires_grad]},
    ]
    '''
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "transformer" not in n
                    and p.requires_grad]},
    ]
    transformer_param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if
                       "transformer" in n and "bias" not in n and p.requires_grad],
            "lr": 1e-4,
            "weight_decay": 1e-2,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       "transformer" in n and "bias" in n and p.requires_grad],
            "lr": 1e-4,
            "weight_decay": 0,
        }
    ]
    optimizer = torch.optim.SGD(
        param_dicts,
        lr=args.base_lr, momentum= args.lr_decay['momentum'], weight_decay= args.lr_decay['weight_decay'])
    base_lrs = [pg['lr'] for pg in optimizer.param_groups] #[0.005] list
    transformer_optimizer = torch.optim.AdamW(transformer_param_dicts, lr=1e-4, weight_decay=1e-4)


    freeze_modules(model, args.freeze_layer)

    time.sleep(2)

    if args.distributed: #false
        # Initialize Process Group
        device = torch.device('cuda', args.local_rank)
        model.to(device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    else:
        model = DataParallel(model, device_ids=device_ids)
        model.to(device)

    # Resume
    #args.resume=True
    if args.resume: #None
        resume_path = None#"/private/5-code/Base_FS_521_4_9/SAMexp/MyFSNet/iSAID/resnet50/split0/5shot/2024-06-02-00-06-24-346119/best.pth"
        #osp.join(args.snapshot_path, args.resume)
        if os.path.isfile(resume_path):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try: 
                model.load_state_dict(new_param)
            except RuntimeError:                   # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
        else:
            if main_process():       
                logger.info("=> no checkpoint found at '{}'".format(resume_path))


    # Get model para.
    total_number, learnable_number = get_model_para_number(model)
    if main_process():
        logger.info('Number of Parameters: %d' % (total_number))
        logger.info('Number of Learnable Parameters: %d' % (learnable_number))

    time.sleep(5)
    return model, optimizer,transformer_optimizer

def main_process():
    return not args.distributed or (args.distributed and (args.local_rank == 0))

def main():
    global device_ids, device
    device_ids = [0,1]
    device = torch.device("cuda:0")

    global args, logger
    args = get_parser()

    args.distributed = False # Debug
    # args.distributed = True if torch.cuda.device_count() > 1 else False
    
    shuffle = False if args.distributed else True

    get_save_path(args)
    logger = get_logger(args.result_path)
    args.logger = logger

    if main_process():
        logger.info(args)

    if args.manual_seed is not None:
        setup_seed(args.manual_seed, args.seed_deterministic)
        
    if args.distributed:
        dist.init_process_group(backend='nccl')
        logger.info('args.local_rank: ', args.local_rank)
        torch.cuda.set_device(args.local_rank)

    if main_process():
        logger.info("=> creating dataset ...")

# ----------------------  DATASET  ----------------------
    #train_data = eval('{}.{}_few_dataset'.format(args.dataset, args.dataset))(split=args.split, \
    #                                        shot=args.shot, mode='train', transform_dict=args.train_transform)
    #iSAID.iSAID_few_dataset
    train_data = eval('{}.{}_few_dataset'.format(args.dataset, args.dataset))(split=args.split, shot=args.shot, mode='train', transform_dict=args.train_transform)
    train_sampler = DistributedSampler(train_data) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=shuffle, \
                                            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    # Val

    val_data = eval('{}.{}_few_dataset'.format(args.dataset, args.dataset))(split=args.split, \
                                            shot=args.shot, mode='val', transform_dict=args.val_transform)            
    val_sampler = DistributedSampler(val_data) if args.distributed else None                  
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, \
                                            num_workers=args.workers, pin_memory=False, sampler=val_sampler)
    logger.info('train_list: {}'.format(train_data.list))
    logger.info('num_train_data: {}'.format(len(train_data)))
    logger.info('val_list: {}'.format(val_data.list))
    logger.info('num_val_data: {}'.format(len(val_data)))

    args.base_class_num = len(train_data.list) #10
    args.novel_class_num = len(val_data.list) #5

    config_file = args.snapshot_path + 'config.yaml'

    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(str(args))
    time.sleep(2)

    if main_process():
        logger.info("=> creating model ...")
        
    model, optimizer,transformer_optimizer = get_model(args,device)
    
    logger.info(model)
    writer = SummaryWriter(args.tb_logger_path)

    # ----------------------  TRAINVAL  ----------------------
    global best_miou, best_FBiou, best_epoch, keep_epoch, val_num
    global best_name, grow_name, all_name, latest_name,all_branch_name, best_class_iou

    best_miou = 0.
    best_FBiou = 0.
    best_class_iou = []
    best_epoch = 0
    keep_epoch = 0
    val_num = 0

    start_time = time.time()
    scaler = amp.GradScaler()

#--------------------------- FilenamePrepare -----------------------------

    latest_name = args.snapshot_path + 'latest.pth'
    best_name = args.snapshot_path + 'best.pth'
    grow_name = args.snapshot_path + 'grow.txt'
    all_name = args.snapshot_path + 'all.txt'
    all_branch_name = args.snapshot_path + 'all_branch.txt'

    for epoch in range(args.start_epoch, args.epochs): #epoch12
        if keep_epoch == args.stop_interval:
            break
        if args.fix_random_seed_val:
            setup_seed(args.manual_seed + epoch, args.seed_deterministic)

        epoch_log = epoch + 1
        keep_epoch += 1

        # ----------------------  TRAIN  ----------------------
        train(train_loader, val_loader, model, optimizer,transformer_optimizer, epoch, scaler, writer)
        torch.cuda.empty_cache()
        # save model for <resuming>
        if ((epoch + 1) % args.save_freq == 0) and main_process():
            epoch_name=args.snapshot_path + str(epoch+1)+'.pth'
            logger.info('Saving checkpoint to: ' + epoch_name)
            #if osp.exists(latest_name):
            #    os.remove(latest_name)
                      
            torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, epoch_name)


        # -----------------------  VAL  -----------------------
        if args.evaluate and (epoch + 1)% args.val_freq == 0: #val_freq=1
            _,fbIou, _,_, mIoU,_ , recall, precision, class_miou,mIou_all = validate(val_loader, model, writer)
            torch.cuda.empty_cache()
            val_num += 1

            with open(all_name, 'a') as f:
                f.write('[{},miou:{:.4f}, fbIou:{:.4f}, recall:{:.4f}, precision:{:.4f},]\n'.format(epoch+1, mIoU, fbIou, recall, precision))
            with open(all_branch_name, 'a') as f:
                f.write('[{},miou---Branch_0:{:.4f}, Branch_1:{:.4f}, Branch_2:{:.4f}, Branch_3:{:.4f},Branch_4:{:.4f}]\n'.format(epoch+1, mIou_all[0], mIou_all[1], mIou_all[2], mIou_all[3],mIou_all[4]))


        # save model for <testing> and <fine-tuning>
            if mIoU > best_miou:
                best_miou, best_epoch, best_class_iou, best_FBiou = mIoU, epoch, class_miou, fbIou
                keep_epoch = 0
                with open(grow_name, 'a') as f:
                    f.write('Best_epoch:{} , Best_miou:{:.4f} , fbIou:{:.4f} , recall:{:.4f}, precision:{:.4f}, \n'.format(epoch+1 , best_miou, fbIou, recall, precision)) 
                logger.info('Saving checkpoint to: ' + best_name + '  miou: {:.4f}'.format(best_miou))
                if osp.exists(best_name):
                    os.remove(best_name)    
                torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, best_name)  

    with open(args.snapshot_path + 'class.txt', 'a') as f:
        for i in range(len(best_class_iou)):
            f.write('{:.4f}\t'.format(best_class_iou[i]))
        f.write('\nmiou: {:.4f}, fb_iou: {:.4f}'.format(best_miou, best_FBiou))
    total_time = time.time() - start_time
    t_m, t_s = divmod(total_time, 60)
    t_h, t_m = divmod(t_m, 60)
    total_time = '{:02d}h {:02d}m {:02d}s'.format(int(t_h), int(t_m), int(t_s))

    logger.info('\nEpoch: {}/{} \t Total running time: {}'.format(epoch_log, args.epochs, total_time))
    logger.info('The number of models validated: {}'.format(val_num))
    logger.info('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  Final Best Result   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    logger.info(args.arch + '\t Group:{} \t Best_mIoU:{:.4f} \t Best_FBIoU:{:.4f} \t Best_step:{}'.format(args.split, best_miou, best_FBiou, best_epoch + 1 ))
    logger.info('>'*80)
    logger.info('Current Date and Time %s' % datetime.datetime.now())


def train(train_loader, val_loader, model, optimizer,transformer_optimizer, epoch,scaler,writer):
    global best_miou, best_epoch, keep_epoch, val_num
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter_1 = AverageMeter()
    aux_loss_meter_2 = AverageMeter()
    aux_loss_meter_3 = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    recall_meter = AverageMeter()
    acc_meter = AverageMeter()
    tmp_num = 0
    model.train()
    if args.fix_bn:
        model.apply(fix_bn) # fix batchnorm

    end = time.time()
    val_time = 0.
    max_iter = args.epochs * len(train_loader) #16212
    
    current_characters = Special_characters[random.randint(0,len(Special_characters)-1)]

    current_GPU = os.environ["CUDA_VISIBLE_DEVICES"]
    # GPU_name = torch.cuda.get_device_name()
    for i, (input, target, s_input, s_mask, padding_mask, s_padding_mask,subcls) in enumerate(train_loader):
        #subcls 选中的类别的索引list_len=5，每个list有8个数  batchsize8 #[6,7,8,9,10,11,12,13,14,15]
        #query(input, target) [8,3,512,512] support( s_input, s_mask)[8,5,3,512,512][8,5,512,512]里面是(0,1,255)
        data_time.update(time.time() - end - val_time)
        current_iter = epoch * len(train_loader) + i + 1
        cur_lr = optimizer.param_groups[0]['lr']
        lr_decay(optimizer, args.base_lr, current_iter, max_iter, args.lr_decay, current_characters )
        if current_iter % 50 == 0 and main_process():
              # optimizer.param_groups[0]['lr']
            logger.info(' ' * len(current_characters[0]) * 3 + ' ' * 10 + 'Base LR: {:.8f}, Curr LR: {:.8f}'.format(
                args.base_lr, cur_lr))
            logger.info(current_characters[0]*3 +' '*5 + '{}_{}_{}_split{}_{}shot Pretrain: {} GPU_id: {}'.format(args.arch,\
                             args.dataset ,args.backbone, args.split, args.shot, args.pretrain, current_GPU) + ' '*5 + current_characters[1]*3)
   
        s_input = s_input.to(device)#.cuda(non_blocking=True)
        s_mask = s_mask.to(device)#.cuda(non_blocking=True)

        padding_mask = padding_mask.to(device)#.cuda(non_blocking=True)
        s_padding_mask = s_padding_mask.to(device)#.cuda(non_blocking=True)
        input = input.to(device)#.cuda(non_blocking=True)
        target = target.to(device)#.cuda(non_blocking=True)
        optimizer.zero_grad()
        transformer_optimizer.zero_grad()
        # with torch.autograd.set_detect_anomaly(True):      # debug
        output,main_loss, aux_loss_1, aux_loss_2,aux_loss_3= model(s_x=s_input, s_y=s_mask, x=input, y=target, padding_mask=padding_mask, s_padding_mask=s_padding_mask, cat_idx=subcls)
        #aux_loss=aux_loss_2 + aux_loss_3 * args.aux_weight1
        #fuse_loss= main_loss + aux_loss_1 * args.aux_weight1
        loss = main_loss + aux_loss_1 * args.aux_weight1+aux_loss_2 + aux_loss_3 * args.aux_weight1
        # fuse_loss+aux_loss
        #print('-------------------======')
        #print(loss) #.mean()
        
        scaler.scale(loss.mean()).backward()
        scaler.step(optimizer)
        scaler.step(transformer_optimizer)
        scaler.update()

        if 'para_limit' in args.keys():
            for item_id in range(len(args.para_limit.name)):
                item = args.para_limit.name[item_id]
                tmp_limit = args.para_limit.limit[item_id]
                eval('model.{}'.format(item)).data.clamp_(tmp_limit[0], tmp_limit[1])

        n = input.size(0) # batch_size

        intersection, union, target = intersectionAndUnionGPU(output, target, 2, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
        
        
        accuracy = sum(intersection_meter.val[1:]) / (sum(target_meter.val[1:]) + 1e-10)  # allAcc
        main_loss_meter.update(main_loss.mean().item(), n)
        if isinstance(aux_loss_1, torch.Tensor):
            aux_loss_meter_1.update(aux_loss_1.mean().item(), n)
        if isinstance(aux_loss_2, torch.Tensor):
            aux_loss_meter_2.update(aux_loss_2.mean().item(), n)
            
        if isinstance(aux_loss_3, torch.Tensor):
            aux_loss_meter_3.update(aux_loss_3.mean().item(), n)
            
        loss_meter.update(loss.mean().item(), n)
        batch_time.update(time.time() - end - val_time)
        end = time.time()

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            # output_masks = output#torch.argmax(torch.softmax(output, dim=1), dim=1, keepdim=True)
            '''
            masks_sam_vis = output.cpu().numpy().astype(np.int16)


            grid_imgs_q = imutils.tensorboard_image_supp(imgs=input.clone())
            # grid_imgs_s = imutils.tensorboard_image_supp(imgs=s_input[:, 0, :, :, :].clone())
            grid_q_mask = imutils.tensorboard_q_mask(imgs=input.clone(), cam=corr_query_mask.detach())  # prompt_mask

            # grid_q_cam_fg = imutils.tensorboard_q_mask(imgs=input.clone(), cam=cam[:, 1, :, :].unsqueeze(1).detach())
            # grid_gts1 = imutils.tensorboard_label(labels=gts1)
            # grid_GT_label_s = imutils.tensorboard_label(labels=s_mask[:, 0, :, :].cpu().numpy().astype(np.int16))
            grid_GT_label_q = imutils.tensorboard_label(labels=target_q.cpu().numpy().astype(np.int16))
            # grid_preds_q = imutils.tensorboard_label(labels=preds)
            # grid_gts_aux = imutils.tensorboard_label(labels=gts_aux)

            grid_preds_SAM_q = imutils.tensorboard_label(labels=masks_sam_vis)
            writer.add_image("train/images_q", grid_imgs_q, global_step=current_iter)
            # writer.add_image("train/images_s", grid_imgs_s, global_step=i)
            writer.add_image("train/query_mask", grid_q_mask, global_step=current_iter)

            # writer.add_image("train/query_cam_fg", grid_q_cam_fg, global_step=i)
            #writer.add_image("cam/valid_cams", grid_cam_q, global_step=current_iter)
            # writer.add_image("cam/gts_aux", grid_gts_aux, global_step=current_iter)
            # writer.add_image("cam/valid_cams_aux", grid_cam_aux_q, global_step=i)
            writer.add_image("train/GT_label_q", grid_GT_label_q, global_step=current_iter)
            # writer.add_image("evl/GT_label_s", grid_GT_label_s, global_step=i)
            # writer.add_image("evl/grid_preds_output_q", grid_preds_output_q, global_step=i)
            # writer.add_image("evl/grid_preds_HQ_q", grid_preds_HQ_q, global_step=i)
            writer.add_image("train/grid_preds_SAM_q", grid_preds_SAM_q, global_step=current_iter)
            '''
            writer.add_scalar('info/lr', cur_lr, current_iter)
            writer.add_scalars('train/MainLoss', {"MainLoss": main_loss_meter.val}, global_step=current_iter)
            writer.add_scalars('train/AuxLoss', {"AuxLoss": aux_loss_meter_1.val}, global_step=current_iter)
            writer.add_scalars('train/AuxLoss2', {"AuxLoss2": aux_loss_meter_2.val}, global_step=current_iter)
            writer.add_scalars('train/AuxLoss3', {"AuxLoss3": aux_loss_meter_3.val}, global_step=current_iter)
            writer.add_scalars('train/Loss', {"Loss": loss_meter.val}, global_step=current_iter)  # similarity_num
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '
                        'AuxLoss_1 {aux_loss_meter_1.val:.4f} '  
                        'AuxLoss_2 {aux_loss_meter_2.val:.4f} ' 
                        'AuxLoss_3 {aux_loss_meter_3.val:.4f} ' 
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                        batch_time=batch_time,
                                                        data_time=data_time,
                                                        remain_time=remain_time,
                                                        main_loss_meter=main_loss_meter,
                                                        aux_loss_meter_1=aux_loss_meter_1,
                                                        aux_loss_meter_2=aux_loss_meter_2,
                                                        aux_loss_meter_3=aux_loss_meter_3,
                                                        loss_meter=loss_meter,
                                                        accuracy=accuracy))


        
        # -----------------------  SubEpoch VAL  -----------------------
        if args.evaluate and args.SubEpoch_val and ((epoch + 1)%args.val_freq==0) and (i in torch.arange(1,args.sub_freq)*round(len(train_loader)/args.sub_freq)): # max_epoch<=100时进行half_epoch Val
            _,fbIou, _,_, mIoU,_ , recall, precision, class_miou,mIou_all = validate(val_loader, model,writer)
            torch.cuda.empty_cache()
            model.train()
            if args.fix_bn:
                model.apply(fix_bn)
            val_num += 1 
            tmp_num += 1
            # save model for <testing> and <fine-tuning>
            sub_epoch_name = args.snapshot_path + str(epoch + 1) + '_1.pth'
            logger.info('Saving checkpoint to: ' + sub_epoch_name)
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       sub_epoch_name)
            with open(all_name, 'a') as f:
                f.write('[{}_{},miou:{:.4f}, fbIou:{:.4f}, recall:{:.4f}, precision:{:.4f},]\n'.format(epoch, tmp_num, mIoU, fbIou, recall, precision))
            with open(all_branch_name, 'a') as f:
                f.write(
                    '[{}_{},miou---Branch_0:{:.4f}, Branch_1:{:.4f}, Branch_2:{:.4f}, Branch_3:{:.4f},Branch_4:{:.4f}]\n'.format(
                        epoch, tmp_num, mIou_all[0], mIou_all[1], mIou_all[2], mIou_all[3], mIou_all[4]))

            if mIoU > best_miou:
                best_miou, best_epoch, best_class_iou, best_FBiou = mIoU, epoch+(1/args.sub_freq)*tmp_num, class_miou, fbIou
                keep_epoch = 0
                with open(grow_name, 'a') as f:
                    f.write('Best_epoch:{}_{} , Best_miou:{:.4f} , fbIou:{:.4f} , recall:{:.4f}, precision:{:.4f}, \n'.format(epoch, tmp_num, best_miou, fbIou, recall, precision)) 
             
                logger.info('Saving checkpoint to: ' + best_name + '  miou: {:.4f}'.format(best_miou))
                if osp.exists(best_name):
                    os.remove(best_name) 
                torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, best_name) 
            
    if hasattr(model, 'out_data'):
        with open(args.snapshot_path + 'out_data.txt', 'a') as f:
            for item in model.out_data:
                f.write(item + '\n')
            model.out_data = []

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    for i in range(2):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))

    return main_loss_meter.avg, mIoU, mAcc, allAcc

def validate(val_loader, model,writer):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    intersection_meter1 = AverageMeter()
    union_meter1 = AverageMeter()
    target_meter1 = AverageMeter()

    intersection_meter2 = AverageMeter()
    union_meter2 = AverageMeter()
    target_meter2 = AverageMeter()

    intersection_meter3 = AverageMeter()
    union_meter3 = AverageMeter()
    target_meter3 = AverageMeter()

    intersection_meter4 = AverageMeter()
    union_meter4 = AverageMeter()
    target_meter4 = AverageMeter()

    split_gap = len(val_loader.dataset.list)
    test_num = 1000  # 20000

    class_intersection_meter = [0] * split_gap
    class_union_meter = [0] * split_gap
    class_target_meter = [0] * split_gap

    class_intersection_meter1 = [0] * split_gap
    class_union_meter1 = [0] * split_gap
    class_target_meter1 = [0] * split_gap

    class_intersection_meter2 = [0] * split_gap
    class_union_meter2 = [0] * split_gap
    class_target_meter2 = [0] * split_gap

    class_intersection_meter3 = [0] * split_gap
    class_union_meter3 = [0] * split_gap
    class_target_meter3 = [0] * split_gap

    class_intersection_meter4 = [0] * split_gap
    class_union_meter4 = [0] * split_gap
    class_target_meter4 = [0] * split_gap

    if args.manual_seed is not None and args.fix_random_seed_val:
        setup_seed(args.manual_seed, args.seed_deterministic)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model.eval()
    end = time.time()
    val_start = end

    assert test_num % args.batch_size_val == 0
    alpha = round(test_num / args.batch_size_val)
    iter_num = 0
    total_time = 0
    for e in range(10):
        for i, (input, target, s_input, s_mask, padding_mask, s_padding_mask, subcls, ori_label) in enumerate(val_loader):
            if iter_num * args.batch_size_val >= test_num:
                break
            iter_num += 1
            data_time.update(time.time() - end)

            s_input = s_input.to(device)#.cuda(non_blocking=True)
            s_mask = s_mask.to(device)#.cuda(non_blocking=True)
            input = input.to(device)#.cuda(non_blocking=True)

            padding_mask = padding_mask.to(device)#.cuda(non_blocking=True)
            s_padding_mask = s_padding_mask.to(device)#.cuda(non_blocking=True)
            target = target.to(device)#.cuda(non_blocking=True)
            ori_label = ori_label.to(device)#.cuda(non_blocking=True)
            start_time = time.time()
            target_test_q = target.clone()
            # with autocast():
            with torch.no_grad():
                output,output_fin_list = model(s_x=s_input, s_y=s_mask, x=input, y=target, padding_mask=padding_mask, s_padding_mask=s_padding_mask,cat_idx=subcls)
            total_time = total_time + 1
            model_time.update(time.time() - start_time)

            if args.ori_resize:
                longerside = max(ori_label.size(1), ori_label.size(2))
                backmask = torch.ones(ori_label.size(0), longerside, longerside, device='cuda')*255
                backmask[:, :ori_label.size(1), :ori_label.size(2)] = ori_label
                target = backmask.clone().long()

            output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
            output = output.float()
            loss = criterion(output, target)

            n = input.size(0)
            loss = torch.mean(loss)

            output = output.max(1)[1] #1,512,512

            # for b_id in range(output.size(0)): target-1,512,512
            intersection, union, new_target = intersectionAndUnionGPU(output, target, 2, args.ignore_label)
            intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)

            tmp_id = subcls[0].cpu().numpy()[0]
            class_intersection_meter[tmp_id] += intersection[1]
            class_union_meter[tmp_id] += union[1]
            class_target_meter[tmp_id] += new_target[1]

            recall = np.mean(intersection_meter.val[1:] / (target_meter.val[1:]+ 1e-10))
            precision = np.mean(intersection_meter.val[1:] /(union_meter.val[1:] - target_meter.val[1:] + intersection_meter.val[1:] + 1e-10) )
            Iou = sum(intersection_meter.val[1:]) / (sum(union_meter.val[1:]) + 1e-10)
            # --------------------------------1111111111111111------------------------
            output1 = F.interpolate(output_fin_list[0], size=target.size()[1:], mode='bilinear',
                                    align_corners=True).float()
            #output1 = output1.max(1)[1]  # 1,512,512
            output1 = nn.Sigmoid()(output1).squeeze(1)
            output1 = torch.where(output1> 0.5, 1, 0)
            intersection1, union1, new_target1 = intersectionAndUnionGPU(output1, target, 2, args.ignore_label)
            intersection1, union1, new_target1 = intersection1.cpu().numpy(), union1.cpu().numpy(), new_target1.cpu().numpy()
            intersection_meter1.update(intersection1), union_meter1.update(union1), target_meter1.update(new_target1)

            tmp_id = subcls[0].cpu().numpy()[0]
            class_intersection_meter1[tmp_id] += intersection1[1]
            class_union_meter1[tmp_id] += union1[1]
            class_target_meter1[tmp_id] += new_target1[1]

            # --------------------------------22222222222------------------------
            output2 = F.interpolate(output_fin_list[1], size=target.size()[1:], mode='bilinear',
                                    align_corners=True).float()
            #output2 = output2.max(1)[1]  # 1,512,512
            output2 = nn.Sigmoid()(output2).squeeze(1)
            output2 = torch.where(output2> 0.5, 1, 0)
            intersection2, union2, new_target2 = intersectionAndUnionGPU(output2, target, 2, args.ignore_label)
            intersection2, union2, new_target2 = intersection2.cpu().numpy(), union2.cpu().numpy(), new_target2.cpu().numpy()
            intersection_meter2.update(intersection2), union_meter2.update(union2), target_meter2.update(new_target2)

            tmp_id = subcls[0].cpu().numpy()[0]
            class_intersection_meter2[tmp_id] += intersection2[1]
            class_union_meter2[tmp_id] += union2[1]
            class_target_meter2[tmp_id] += new_target2[1]

            # --------------------------------33333333333333333333------------------------
            output3 = F.interpolate(output_fin_list[2], size=target.size()[1:], mode='bilinear',
                                    align_corners=True).float()
            #output3 = output3.max(1)[1]  # 1,512,512
            output3 = nn.Sigmoid()(output3).squeeze(1)
            output3 = torch.where(output3> 0.5, 1, 0)
            intersection3, union3, new_target3 = intersectionAndUnionGPU(output3, target, 2, args.ignore_label)
            intersection3, union3, new_target3 = intersection3.cpu().numpy(), union3.cpu().numpy(), new_target3.cpu().numpy()
            intersection_meter3.update(intersection3), union_meter3.update(union3), target_meter3.update(new_target3)

            tmp_id = subcls[0].cpu().numpy()[0]
            class_intersection_meter3[tmp_id] += intersection3[1]
            class_union_meter3[tmp_id] += union3[1]
            class_target_meter3[tmp_id] += new_target3[1]

            # --------------------------------444444444444444444-----------------------
            output4 = F.interpolate(output_fin_list[3], size=target.size()[1:], mode='bilinear',
                                    align_corners=True).float()
            #output4 = output4.max(1)[1]  # 1,512,512
            output4 = nn.Sigmoid()(output4).squeeze(1)
            output4 = torch.where(output4> 0.5, 1, 0)
            intersection4, union4, new_target4 = intersectionAndUnionGPU(output4, target, 2, args.ignore_label)
            intersection4, union4, new_target4 = intersection4.cpu().numpy(), union4.cpu().numpy(), new_target4.cpu().numpy()
            intersection_meter4.update(intersection4), union_meter4.update(union4), target_meter4.update(new_target4)

            tmp_id = subcls[0].cpu().numpy()[0]
            class_intersection_meter4[tmp_id] += intersection4[1]
            class_union_meter4[tmp_id] += union4[1]
            class_target_meter4[tmp_id] += new_target4[1]

            loss_meter.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if ((iter_num) % round((alpha/20)) == 0):

                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'recall {recall:.4f} '
                            'precision {precision:.4f} '
                            'Iou {Iou:.4f}.'.format(iter_num* args.batch_size_val, test_num,
                                                            data_time=data_time,
                                                            batch_time=batch_time,
                                                            loss_meter=loss_meter,
                                                            recall=recall,
                                                            precision=precision,
                                                            Iou=Iou))
    val_time = time.time()-val_start
    mIou_all = []
    iou_class0 = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class0 = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU0 = np.mean(iou_class0)
    mAcc0 = np.mean(accuracy_class0)
    allAcc0 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)


    class_iou_class0 = []
    class_miou0 = 0
    class_recall_class0 = []
    class_mrecall0 = 0
    class_precisoin_class0 = []
    class_mprecision0 = 0

    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i]/(class_union_meter[i]+ 1e-10)
        class_iou_class0.append(class_iou)
        class_miou0 += class_iou

        class_recall = class_intersection_meter[i]/(class_target_meter[i]+ 1e-10)
        class_recall_class0.append(class_recall)
        class_mrecall0 += class_recall

        class_precision = class_intersection_meter[i]/(class_union_meter[i] - class_target_meter[i] + class_intersection_meter[i]+ 1e-10)
        class_precisoin_class0.append(class_precision)
        class_mprecision0 += class_precision

    class_mrecall0 = class_mrecall0*1.0 / len(class_intersection_meter)
    class_miou0 = class_miou0*1.0 / len(class_intersection_meter)
    class_mprecision0 = class_mprecision0*1.0 / len(class_intersection_meter)


    logger.info('mean IoU---Val result: mIoU {:.4f}.'.format(class_miou0))
    logger.info('mean recall---Val result: mrecall {:.4f}.'.format(class_mrecall0))
    logger.info('mean precisoin---Val result: mprecisoin {:.4f}.'.format(class_mprecision0))

    for i in range(split_gap):
        logger.info('Class_{}: \t Result: iou {:.4f}. \t recall {:.4f}. \t precision {:.4f}. \t {}'.format(i+1, \
                        class_iou_class0[i], class_recall_class0[i], class_precisoin_class0[i],\
                         val_loader.dataset.class_id[val_loader.dataset.list[i]]))



    logger.info('FBIoU---Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU0, mAcc0, allAcc0))
    for i in range(2):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class0[i], accuracy_class0[i]))

    mIou_all.append(class_miou0)



    logger.info('------------------------------Branch----1111111111111-------------------------------')
    intersection_meter=intersection_meter1
    union_meter=union_meter1
    target_meter=target_meter1
    class_intersection_meter=class_intersection_meter1
    class_union_meter=class_union_meter1
    class_target_meter=class_target_meter1

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    class_iou_class = []
    class_miou = 0
    class_recall_class = []
    class_mrecall = 0
    class_precisoin_class = []
    class_mprecision = 0

    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i] / (class_union_meter[i] + 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou

        class_recall = class_intersection_meter[i] / (class_target_meter[i] + 1e-10)
        class_recall_class.append(class_recall)
        class_mrecall += class_recall

        class_precision = class_intersection_meter[i] / (
                    class_union_meter[i] - class_target_meter[i] + class_intersection_meter[i] + 1e-10)
        class_precisoin_class.append(class_precision)
        class_mprecision += class_precision

    class_mrecall = class_mrecall * 1.0 / len(class_intersection_meter)
    class_miou = class_miou * 1.0 / len(class_intersection_meter)
    class_mprecision = class_mprecision * 1.0 / len(class_intersection_meter)

    logger.info('mean IoU---Val result: mIoU {:.4f}.'.format(class_miou))
    logger.info('mean recall---Val result: mrecall {:.4f}.'.format(class_mrecall))
    logger.info('mean precisoin---Val result: mprecisoin {:.4f}.'.format(class_mprecision))

    for i in range(split_gap):
        logger.info('Class_{}: \t Result: iou {:.4f}. \t recall {:.4f}. \t precision {:.4f}. \t {}'.format(i + 1, \
                                                                                                           class_iou_class[
                                                                                                               i],
                                                                                                           class_recall_class[
                                                                                                               i],
                                                                                                           class_precisoin_class[
                                                                                                               i], \
                                                                                                           val_loader.dataset.class_id[
                                                                                                               val_loader.dataset.list[
                                                                                                                   i]]))
    logger.info('FBIoU---Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(2):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))

    mIou_all.append(class_miou)
    logger.info('------------------------------Branch----22222222222222-------------------------------')
    intersection_meter = intersection_meter2
    union_meter = union_meter2
    target_meter = target_meter2
    class_intersection_meter = class_intersection_meter2
    class_union_meter = class_union_meter2
    class_target_meter = class_target_meter2


    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    class_iou_class = []
    class_miou = 0
    class_recall_class = []
    class_mrecall = 0
    class_precisoin_class = []
    class_mprecision = 0

    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i] / (class_union_meter[i] + 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou

        class_recall = class_intersection_meter[i] / (class_target_meter[i] + 1e-10)
        class_recall_class.append(class_recall)
        class_mrecall += class_recall

        class_precision = class_intersection_meter[i] / (
                class_union_meter[i] - class_target_meter[i] + class_intersection_meter[i] + 1e-10)
        class_precisoin_class.append(class_precision)
        class_mprecision += class_precision

    class_mrecall = class_mrecall * 1.0 / len(class_intersection_meter)
    class_miou = class_miou * 1.0 / len(class_intersection_meter)
    class_mprecision = class_mprecision * 1.0 / len(class_intersection_meter)

    logger.info('mean IoU---Val result: mIoU {:.4f}.'.format(class_miou))
    logger.info('mean recall---Val result: mrecall {:.4f}.'.format(class_mrecall))
    logger.info('mean precisoin---Val result: mprecisoin {:.4f}.'.format(class_mprecision))

    for i in range(split_gap):
        logger.info('Class_{}: \t Result: iou {:.4f}. \t recall {:.4f}. \t precision {:.4f}. \t {}'.format(i + 1, \
                                                                                                           class_iou_class[
                                                                                                               i],
                                                                                                           class_recall_class[
                                                                                                               i],
                                                                                                           class_precisoin_class[
                                                                                                               i], \
                                                                                                           val_loader.dataset.class_id[
                                                                                                               val_loader.dataset.list[
                                                                                                                   i]]))
    logger.info('FBIoU---Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(2):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    mIou_all.append(class_miou)
    logger.info('------------------------------Branch----33333333333333333333-------------------------------')
    intersection_meter = intersection_meter3
    union_meter = union_meter3
    target_meter = target_meter3
    class_intersection_meter = class_intersection_meter3
    class_union_meter = class_union_meter3
    class_target_meter = class_target_meter3


    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    class_iou_class = []
    class_miou = 0
    class_recall_class = []
    class_mrecall = 0
    class_precisoin_class = []
    class_mprecision = 0

    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i] / (class_union_meter[i] + 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou

        class_recall = class_intersection_meter[i] / (class_target_meter[i] + 1e-10)
        class_recall_class.append(class_recall)
        class_mrecall += class_recall

        class_precision = class_intersection_meter[i] / (
                class_union_meter[i] - class_target_meter[i] + class_intersection_meter[i] + 1e-10)
        class_precisoin_class.append(class_precision)
        class_mprecision += class_precision

    class_mrecall = class_mrecall * 1.0 / len(class_intersection_meter)
    class_miou = class_miou * 1.0 / len(class_intersection_meter)
    class_mprecision = class_mprecision * 1.0 / len(class_intersection_meter)

    logger.info('mean IoU---Val result: mIoU {:.4f}.'.format(class_miou))
    logger.info('mean recall---Val result: mrecall {:.4f}.'.format(class_mrecall))
    logger.info('mean precisoin---Val result: mprecisoin {:.4f}.'.format(class_mprecision))

    for i in range(split_gap):
        logger.info('Class_{}: \t Result: iou {:.4f}. \t recall {:.4f}. \t precision {:.4f}. \t {}'.format(i + 1, \
                                                                                                           class_iou_class[
                                                                                                               i],
                                                                                                           class_recall_class[
                                                                                                               i],
                                                                                                           class_precisoin_class[
                                                                                                               i], \
                                                                                                           val_loader.dataset.class_id[
                                                                                                               val_loader.dataset.list[
                                                                                                                   i]]))
    logger.info('FBIoU---Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(2):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    mIou_all.append(class_miou)
    logger.info('------------------------------Branch----4444444444444444444-------------------------------')
    intersection_meter = intersection_meter4
    union_meter = union_meter4
    target_meter = target_meter4
    class_intersection_meter = class_intersection_meter4
    class_union_meter = class_union_meter4
    class_target_meter = class_target_meter4

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    class_iou_class = []
    class_miou = 0
    class_recall_class = []
    class_mrecall = 0
    class_precisoin_class = []
    class_mprecision = 0

    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i] / (class_union_meter[i] + 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou

        class_recall = class_intersection_meter[i] / (class_target_meter[i] + 1e-10)
        class_recall_class.append(class_recall)
        class_mrecall += class_recall

        class_precision = class_intersection_meter[i] / (
                class_union_meter[i] - class_target_meter[i] + class_intersection_meter[i] + 1e-10)
        class_precisoin_class.append(class_precision)
        class_mprecision += class_precision

    class_mrecall = class_mrecall * 1.0 / len(class_intersection_meter)
    class_miou = class_miou * 1.0 / len(class_intersection_meter)
    class_mprecision = class_mprecision * 1.0 / len(class_intersection_meter)

    logger.info('mean IoU---Val result: mIoU {:.4f}.'.format(class_miou))
    logger.info('mean recall---Val result: mrecall {:.4f}.'.format(class_mrecall))
    logger.info('mean precisoin---Val result: mprecisoin {:.4f}.'.format(class_mprecision))

    for i in range(split_gap):
        logger.info('Class_{}: \t Result: iou {:.4f}. \t recall {:.4f}. \t precision {:.4f}. \t {}'.format(i + 1, \
                                                                                                           class_iou_class[
                                                                                                               i],
                                                                                                           class_recall_class[
                                                                                                               i],
                                                                                                           class_precisoin_class[
                                                                                                               i], \
                                                                                                           val_loader.dataset.class_id[
                                                                                                               val_loader.dataset.list[
                                                                                                                   i]]))
    logger.info('FBIoU---Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(2):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))

    mIou_all.append(class_miou)

    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    logger.info('total time: {:.4f}, avg inference time: {:.4f}, count: {}'.format(val_time, model_time.avg, test_num))

    return loss_meter.avg, mIoU0, mAcc0, allAcc0, class_miou0, iou_class0[1], class_mrecall0, class_mprecision0, class_iou_class0,mIou_all


if __name__ == '__main__':
    main()
