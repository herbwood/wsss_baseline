import os
import random 
from PIL import Image
import numpy as np

import torch
import torchvision 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class VOCDataset(Dataset):
    def __init__(self, datalist_file, root_dir, num_classes=20, transform=None):
        self.datalist_file = datalist_file
        self.root_dir = root_dir 
        self.num_classes = num_classes 
        self.transform = transform 
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(img_name).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
            
        if self.testing:
            return img_name, image, self.label_list[idx]
        
        return image, self.label_list[idx]
    
    def read_labeled_image_list(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        
        img_name_list = []
        img_labels = []
        
        for line in lines:
            fields = line.strip().split() # image_name, label0, label1, ... 
            image = fields[0] + '.jpg'
            labels = np.zeros((self.num_classes,), dtype=np.float32) # [0, 0, ..., 0]
            
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
            
        return img_name_list, img_labels  


def train_data_loader(args):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
        
    input_size = int(args.input_size)
    
    transform_train = transforms.Compose([transforms.Resize(input_size),
                                          transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean_vals, std_vals)
                                          ])
    
    train_dataset = VOCDataset(args.train_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=args.bach_size, shuffle=True, num_workers=args.num_workers)
    
    return train_loader


def test_data_loader(args):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
        
    input_size = int(args.input_size)

    transform_test = transforms.Compose([transforms.Resize(input_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean_vals, std_vals)
                                         ])
    
    val_dataset = VOCDataset(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=transform_test)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    return val_loader