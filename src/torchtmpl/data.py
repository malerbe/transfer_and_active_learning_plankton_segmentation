# coding: utf-8

# Standard imports
import logging
import random

# External imports
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class PelgasDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image introuvable : {path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label


def show_image(X):
    num_c = X.shape[0]
    plt.figure()
    plt.imshow(X[0] if num_c == 1 else X.permute(1, 2, 0))
    plt.show()

def get_train_transforms(data_config):
    resize = data_config["resize"]

    return A.Compose([
        A.Resize(height=resize, width=resize),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, p=0.7, border_mode=cv2.BORDER_CONSTANT),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),

        A.CoarseDropout(
            num_holes_range=(2, 15),    
            hole_height_range=(2, 10),  
            hole_width_range=(2, 10),    
            fill_value=0, 
            mask_fill_value=None, 
            p=0.6
        ),

        A.Downscale(scale_range=(0.5, 0.9), p=0.4),
        
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=0.5),   
            A.GaussianBlur(blur_limit=(3, 5), p=0.5), 
        ], p=0.3),
        
        A.OneOf([
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=0.5),
        ], p=0.5),

        A.ImageCompression(quality_range=(40, 80), p=0.4),

        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        ToTensorV2() 
    ])

def get_valid_transforms(data_config):
    resize = data_config["resize"]

    return A.Compose([
        A.Resize(width=resize, height=resize),
        A.ToGray(p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])



def get_dataloaders(data_config, use_cuda):
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]
    resize = data_config["resize"]

    logging.info("  - Dataset creation")

    raw_dataset = ImageFolder(root=data_config["trainpath"])

    all_samples = raw_dataset.samples
    classes = raw_dataset.classes

    random.seed(42) 
    random.shuffle(all_samples)

    num_valid = int(valid_ratio * len(all_samples))
    train_samples = all_samples[num_valid:]
    valid_samples = all_samples[:num_valid]

    train_ds = PelgasDataset(
        samples = train_samples, 
        transform=get_train_transforms(data_config)
    )

    valid_ds = PelgasDataset(
        samples = valid_samples, 
        transform=get_valid_transforms(data_config)
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )


    logging.info(f"  - Train size: {len(train_ds)}, Valid size: {len(valid_ds)}")

    

    num_classes = len(classes)
    input_size = (1, resize, resize)

    return train_loader, valid_loader, input_size, num_classes
