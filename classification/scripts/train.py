import os
import sys
sys.path.append('/home/junehyoung/wsss_baseline/classification')
import time 
import shutil 
import argparse 
import wandb 

import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 

from models import vgg
from util.pyutils import AverageMeter 
from util.torchutils import reduce_lr
from util.LoadData import train_data_loader 


def get_arguments():
    parser = argparse.ArgumentParser(description='The pytorch implementation of Weakly Supervised Semantic Segmentation')
    
    parser.add_argument("--img_dir", type=str, default='', help='Directory of training images')
    parser.add_argument("--snapshot_dir", type=str, default='', help='training image path')
    parser.add_argument("--train_list", type=str, default='None', help='train list path')
    parser.add_argument("--test_list", type=str, default='None', help='test list path')
    parser.add_argument("--att_dir", type=str, default='./runs/', help='attention map save path')
    
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.6)
    
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--decay_points", type=str, default='61')
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--resume", type=str, default='False')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--current_epoch", type=int, default=0)
    parser.add_argument("--wandb_name", type=str, default='')

    return parser.parse_args()

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savepath = os.path.join(args.snapshot_dir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))
        
def get_model(args):
    model = vgg.vgg16(pretrained=True, num_classes=args.num_classes, att_dir=args.att_dir, training_epoch=args.epoch)
    model = torch.nn.DataParallel(model).cuda()
    param_groups = model.module.get_parameter_groups()
    
    # specify per-layer learning rates
    optimizer = optim.SGD([
        {'params' : param_groups[0], 'lr' : args.lr},
        {'params' : param_groups[1], 'lr' : 2 * args.lr},
        {'params' : param_groups[2], 'lr' : 10 * args.lr},
        {'params' : param_groups[3], 'lr' : 20 * args.lr},
        ],
        momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    
    return model, optimizer 

def train(args):

    batch_time = AverageMeter()
    losses = AverageMeter()
    
    total_epoch = args.epoch 
    global_counter = args.global_counter 
    current_epoch = args.current_epoch 
    
    train_loader = train_data_loader(args)
    max_step = total_epoch * len(train_loader)
    args.max_step = max_step 
    print(f'Max step : {max_step}')
    
    model, optimizer = get_model(args)
    print(model)
    model.train()
    end = time.time() 

    wandb.init()
    wandb.run.name = args.wandb_name 
    wandb.config.update(args)
    wandb.watch(model)
    
    while current_epoch < total_epoch:
        
        # reset loss and batch time when epoch starts 
        model.train()
        losses.reset()
        batch_time.reset()
        
        res = reduce_lr(args, optimizer, current_epoch) # reduce learning rate 
        steps_per_epoch = len(train_loader)
        
        index = 0
        # example_images = []
        for idx, dat in enumerate(train_loader):
            
            img_name, img, label = dat 
            label = label.cuda(non_blocking=True)
            
            logits = model(img, current_epoch, label, img_name)
            index += args.batch_size 
            
            if len(logits.shape) == 1:
                logits = logits.reshape(label.shape)
            loss_val = F.multilabel_soft_margin_loss(logits, label)

            # example_images.append(wandb.Image(img[0]))
            
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
            losses.update(loss_val.data.item(), img.size()[0]) # (loss value, batch size)
            batch_time.update(time.time() - end)
            end = time.time()
            
            # global_counter counts number of batches 
            #  if number of batch hits 1000, reset losses 
            global_counter += 1
            # if global_counter % 1000 == 0:
            #     losses.reset()
            
            if global_counter % args.disp_interval == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'LR: {:.5f}\t' 
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        current_epoch, global_counter%len(train_loader), len(train_loader), 
                        optimizer.param_groups[0]['lr'], loss=losses))
                wandb.log({'Train Loss' : losses.val,
                           'Accumulated Train Loss' : losses.avg,
                        #    'Examples' : example_images
                         })
                
        if current_epoch == args.epoch - 1:
            save_checkpoint(args, 
                            {
                                'epoch' : current_epoch, 
                                'global_counter' : global_counter, 
                                'state_dict' : model.state_dict(),
                                'optimizer' : optimizer.state_dict() 
                            }, is_best=False,
                            filename=f'{args.dataset}_epoch_{current_epoch}.pth')
        current_epoch += 1    

if __name__ == '__main__':
    args = get_arguments()
    
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    train(args)