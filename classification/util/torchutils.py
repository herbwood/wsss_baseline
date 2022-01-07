import os  
import torch 
import torch.optim as optim


def reduce_lr(args, optimizer, epoch, factor=0.1):
    values = args.decay_points.strip().split(',')
    try:
        change_points = map(lambda x : int(x.strip()), values)
    except:
        change_points = None 
        
    if change_points is not None and epoch in change_points:
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] * factor 
            print(epoch, g['lr'])
        
        return True 
    
def restore(args, model):
    if os.path.isfile(args.restore_from) and ('.pth' in args.restore_from):
        snapshot = args.restore_from 
    # if tar file 
    else:
        restore_dir = args.snapshot_dir 
        filelist = os.listdir(restore_dir)
        filelist = [x for x in filelist if os.path.isfile(os.path.join(restore_dir, x)) and x.endswith('.pth.tar')]
        if len(filelist) > 0:
            filelist.sort(key=lambda fn: os.path.getmtime(os.path.join(restore_dir, fn)), reverse=True)
            snapshot = os.path.join(restore_dir, filelist[0])
        else:
            snapshot = ''
    
    if os.path.isfile(snapshot):
        print(f"=> loading checkpoint '{snapshot}'")
        checkpoint = torch.load(snapshot)
        _model_load(model, checkpoint)
        print(f"=> loaded checkpoint '{snapshot}'")
    else:
        print(f"=> no checkpoint found at '{snapshot}'")
        
def _model_load(model, pretrained_dict):
    model_dict = model.state_dict()
    
    if list(model_dict.keys())[0].startswith('module.'):
        pretrained_dict = {'module.'+k: v for k, v in pretrained_dict.items()}
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    # print('Weights cannot be loaded: ')
    # print([k for k in model_dict.keys() if k not in pretrained_dict.keys()])
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)    
    