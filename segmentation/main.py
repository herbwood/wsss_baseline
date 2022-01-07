import os
import json 
import multiprocessing 
import click 
import joblib 
from tqdm import tqdm 
import numpy as np 
from PIL import Image 
from omegaconf import OmegaConf 

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchnet.meter import MovingAverageValueMeter

from libs.datasets import get_dataset 
from libs.models import DeepLabV2_ResNet101_MSC 
from libs.utils import DenseCRF, PolynomialLR, scores 


# TODO
# move functions to util directory 
def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    if cuda:
        print("Device:")
        for i in range(torch.cuda.device_count()):
            print(f"    {i}", torch.cuda.get_device_name(i))
    else:
        print("Device: CPU")

    return device 

def get_params(model, key):
    # for Dilated FCN 
    if key == '1x':
        for m in model.named_modules():
            if 'layer' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p 

    # for conv weight in the ASPP module 
    if key == "10x":
        for m in model.named_children():
            if 'aspp' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].weight 
    
    # for conv bias in the ASPP module 
    if key == "20x":
        for m in model.named_modules():
            if 'aspp' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].bias 

def resize_labels(labels, size):
    new_labels = []
    for label in labels:
        label = label.float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    new_labels = torch.LongTensor(new_labels)

    return new_labels 

# TODO
# comment all these blocks 
@click.group()
@click.pass_context
def main(ctx):
    """
    Training and evaluation
    """
    print("Mode:", ctx.invoked_subcommand)


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "--cuda/--cpu", default=True, help="Enable CUDA if available [default: --cuda]"
)

def train(config_path, cuda):

    ###### Confiuguration 
    CONFIG = OmegaConf.load(config_path)
    device = get_device(cuda)
    torch.backends.cudnn.benchmark = True 

    ###### Dataset 
    dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.TRAIN,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=True,
        base_size=CONFIG.IMAGE.SIZE.BASE,
        crop_size=CONFIG.IMAGE.SIZE.TRAIN,
        scales=CONFIG.DATASET.SCALES,
        flip=True,
    )
    print(dataset) 

    ##### DataLoader 
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TRAIN,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=True,
    )
    loader_iter = iter(loader)

    ##### Model check 
    # TODO 
    # more support on other segmentation models 
    print(f"Model: {CONFIG.MODEL.NAME}")
    assert (
        CONFIG.MODEL.NAME == "DeepLabV2_ResNet101_MSC"
    ), 'Currently support only "DeepLabV2_ResNet101_MSC"'

    ##### Model setup 
    model = DeepLabV2_ResNet101_MSC(n_classes=CONFIG.DATASET.N_CLASSES)
    # initialize backbone network 
    state_dict = torch.load(CONFIG.MODEL.INIT_MODEL)
    print(f"    Init: {CONFIG.MODEL.INIT_MODEL}")

    for m in model.base.state_dict().keys():
        if m not in state_dict.keys():
            print(f"    Skip init: {m}")
    model.base.load_state_dict(state_dict, strict=False) # to skip ASPP 
    model = nn.DataParallel(model)
    model.to(device)

    ##### Loss function definition
    criterion = nn.CrossEntropyLoss(ignore_index=CONFIG.DATASET.IGNORE_LABEL)
    criterion.to(device)

    ##### optimizer 
    optimizer = torch.optim.SGD(
        params=[
            {
                "params": get_params(model.module, key="1x"),
                "lr": CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="10x"),
                "lr": 10 * CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="20x"),
                "lr": 20 * CONFIG.SOLVER.LR,
                "weight_decay": 0.0,
            },
        ],
        momentum=CONFIG.SOLVER.MOMENTUM,
    )

    ##### Learning rate scheduler 
    scheduler = PolynomialLR(
        optimizer=optimizer,
        step_size=CONFIG.SOLVER.LR_DECAY,
        iter_max=CONFIG.SOLVER.ITER_MAX,
        power=CONFIG.SOLVER.POLY_POWER,
    )

    ##### Setup loss logger 
    # TODO
    # 1) add averagemeter 
    # 2) add wandb 
    writer = SummaryWriter(os.path.join(CONFIG.EXP.OUTPUT_DIR, "logs", CONFIG.EXP.ID))
    average_loss = MovingAverageValueMeter(CONFIG.SOLVER.AVERAGE_LOSS)

    ##### Path to save models
    checkpoint_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "models",
        CONFIG.EXP.ID,
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.TRAIN,
    )
    makedirs(checkpoint_dir)
    print("Checkpoint dst:", checkpoint_dir)

    ##### Train the model 
    # freeze the batch norm pre-trained on COCO
    model.train()
    model.module.base.freeze_bn()

    for iteration in tqdm(
        range(1, CONFIG.SOLVER.ITER_MAX + 1),
        total=CONFIG.SOLVER.ITER_MAX,
        dynamic_ncols=True,
    ):

        optimizer.zero_grad()

        loss = 0
        for _ in range(CONFIG.SOLVER.ITER_SIZE):
            try:
                _, images, labels = next(loader_iter)
            except:
                loader_iter = iter(loader)
                _, images, labels = next(loader_iter)
            
            logits = model(images.to(device))

            iter_loss = 0
            for logit in logits:
                # resize labels for {100%, 75%, 50%, MAX} logits 
                _, _, H, W = logit.shape 
                labels_ = resize_labels(labels, size=(H, W))
                iter_loss += criterion(logit, labels_.to(device))

            iter_loss /= CONFIG.SOLVER.ITER_SIZE 
            iter_loss.backward()
            loss += float(iter_loss)
        
        average_loss.add(loss)

        optimizer.step()
        scheduler.step(epoch=iteration)

        # Tensorboard 
        if iteration % CONFIG.SOLVER.ITER_TB == 0:
            writer.add_scalar("loss/train", average_loss.value()[0], iteration)
            for i, o in enumerate(optimizer.param_groups):
                writer.add_scalar("lr/group_{}".format(i), o["lr"], iteration)
            for i in range(torch.cuda.device_count()):
                writer.add_scalar(
                    "gpu/device_{}/memory_cached".format(i),
                    torch.cuda.memory_cached(i) / 1024 ** 3,
                    iteration,
                )

            if False:
                for name, param in model.module.base.named_parameters():
                    name = name.replace(".", "/")
                    # Weight/gradient distribution
                    writer.add_histogram(name, param, iteration, bins="auto")
                    if param.requires_grad:
                        writer.add_histogram(
                            name + "/grad", param.grad, iteration, bins="auto"
                        )
        
        # Save a model 
        if iteration % CONFIG.SOLVER.ITER_SAVE == 0:
            torch.save(
                model.module.state_dict(),
                os.path.join(checkpoint_dir, f"checkpoint_{iteration}.pth")
            )
    
    torch.save(
        model.module.state_dict, 
        os.path.join(checkpoint_dir, "checkpoint_final.pth")
    )