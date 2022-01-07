import os
import sys
sys.path.append('/home/junehyoung/wsss_baseline/classification')
import argparse 
import cv2
import numpy as np 
from tqdm import tqdm 

import torch 
import torch.nn.functional as F 

from models import vgg
from util.LoadData import test_data_loader 


def make_cam(x, epsilon=1e-5):
    # relu(x) = max(x, 0)
    x = F.relu(x)
    
    b, c, h, w = x.size()

    flat_x = x.view(b, c, (h * w))
    max_value = flat_x.max(axis=-1)[0].view((b, c, 1, 1))
    
    return F.relu(x - epsilon) / (max_value + epsilon)

def get_numpy_from_tensor(tensor):
    return tensor.cpu().detach().numpy()

def transpose(image):
    return image.transpose((1, 2, 0))

def denormalize(image, mean=None, std=None, dtype=np.uint8, tp=True):
    if tp:
        image = transpose(image)
        
    if mean is not None:
        image = (image * std) + mean
    
    if dtype == np.uint8:
        image *= 255.
        return image.astype(np.uint8)
    else:
        return image

def colormap(cam, shape=None, mode=cv2.COLORMAP_JET):
    if shape is not None:
        h, w, c = shape
        cam = cv2.resize(cam, (w, h))
    cam = cv2.applyColorMap(cam, mode)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

    return cam

def get_arguments():
    parser = argparse.ArgumentParser(description='CAM visualization')
    
    parser.add_argument("--save_dir", type=str, default=f'../runs/{EXP_NAME}/cam/')
    parser.add_argument("--img_dir", type=str, default='../../../dataset/VOC2012/JPEGImages/')
    parser.add_argument("--test_list", type=str, default='../data/train_cls.txt')
    parser.add_argument("--restore_from", type=str, default=f'../runs/{EXP_NAME}/model/pascal_voc_epoch_14.pth')
    
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--dataset", type=str, default='voc2012')
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--arch", type=str,default='vgg_v0')
    
    return parser.parse_args()

def get_model(args):
    model = vgg.vgg16(num_classes=args.num_classes)
    model = torch.nn.DataParallel(model).cuda()
    
    pretrained_dict = torch.load(args.restore_from)['state_dict']
    model_dict = model.state_dict()
    
    # print(model_dict.keys())
    # print(pretrained_dict.keys())
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    # print("Weights cannot be loaded:")
    # print([k for k in model_dict.keys() if k not in pretrained_dict.keys()])

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model
    
def visualize_cams(args):
    print('\nValidating ...', flush=True, end='')
    
    model = get_model(args)
    model.eval()
    val_loader = test_data_loader(args)
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    with torch.no_grad():
        for idx, dat in tqdm(enumerate(val_loader)):
            img_name, img, label_in = dat
            label = label_in.cuda(non_blocking=True)
            logits = model(img)
            last_featmaps = model.module.get_heatmaps()

            ###################CAM Visualization#########################

            mask = label.unsqueeze(2).unsqueeze(3)
            cams = (make_cam(last_featmaps) * mask)
            # print(cams.shape)
            # cams[:, 6, :, :] = 0
            obj_cams = cams.max(dim=1)[0]
            
            for b in range(args.batch_size):
                image = get_numpy_from_tensor(img[b])
                cam = get_numpy_from_tensor(obj_cams[b])

                image = denormalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])[..., ::-1]
                h, w, c = image.shape

                cam = (cam * 255).astype(np.uint8)
                cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
                cam = colormap(cam)

                image = cv2.addWeighted(image, 0.5, cam, 0.5, 0)[..., ::-1]
                image = image.astype(np.float32) / 255.
                filename = img_name[0].split('/')[-1]
                cv2.imwrite(args.save_dir + filename, image*255)
                
if __name__ == "__main__":
    EXP_NAME = 'naive_cam'
    args = get_arguments()
    visualize_cams(args)    