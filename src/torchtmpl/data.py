# coding: utf-8

# Standard imports
import logging
import random
import os

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
        img_path, mask_path = self.samples[idx] 
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image introuvable : {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) 
        if mask is None:
             raise FileNotFoundError(f"Masque introuvable : {mask_path}")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask


def show_image(X):
    num_c = X.shape[0]
    plt.figure()
    plt.imshow(X[0] if num_c == 1 else X.permute(1, 2, 0))
    plt.show()

def get_train_transforms(data_config):
    resize = data_config["resize"]

    return A.Compose([
        # --- 1. GÉOMÉTRIE ---
        A.Resize(width=resize, height=resize),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, p=0.7, border_mode=cv2.BORDER_CONSTANT), 
        
        # Déformation élastique légère (simule le mouvement de l'eau/corps mous)
        A.ElasticTransform(alpha=1, sigma=30, p=0.3), 

        # --- 2. ARTEFACTS (NOUVEAU) ---
        # Simule des poussières / saletés sur l'objectif ou dans l'eau
        # On crée entre 1 et 15 petits points noirs (ou gris foncé)
        A.CoarseDropout(
            max_holes=8,       # Max nb de poussières
            min_holes=2,        # Min nb de poussières
            max_height=8,      # Taille max (pixels)
            max_width=8, 
            min_height=2,       # Taille min (pixels)
            min_width=2,
            fill_value=0,       # 0 = Noir (poussière opaque), ou mettre 'random'
            mask_fill_value=None, # Ne touche pas au masque ! (IMPORTANT)
            p=0.6
        ),

        # --- 3. DÉGRADATION & BRUIT ---
        
        # Réduction de résolution (simule une optique bas de gamme)
        A.Downscale(scale_range=(0.8, 0.95), p=0.4),
        
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.5),   
            A.GaussianBlur(blur_limit=(3, 3), p=0.5), 
        ], p=0.3),
        
        # Bruit de capteur (Remplacement de GaussNoise pour éviter l'erreur)
        A.OneOf([
            # Simulation de bruit ISO (grain photo) - très réaliste
            A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.05, 0.3), p=0.5),
            
            # Bruit multiplicatif (variation de sensibilité locale)
            A.MultiplicativeNoise(multiplier=(0.95, 1.05), elementwise=True, p=0.5),
        ], p=0.5),

        # Compression JPEG (Artefacts de blocs carrés)
        A.ImageCompression(quality_range=(70, 95), p=0.4),

        # --- 4. LUMIÈRE ---
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        
        # --- 5. FINITION : FORCER LE GRIS ---
        # Convertit tout (y compris le bruit coloré ISO) en gris pur
        A.ToGray(p=1.0),
    
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        ToTensorV2() 
    ])

def get_valid_transforms(data_config):
    resize = data_config["resize"]

    return A.Compose([
        A.Resize(width=resize, height=resize),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToGray(p=1.0),
        ToTensorV2()
    ])



def get_dataloaders(data_config, use_cuda):
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]
    resize = data_config["resize"]
    root_dir = data_config["trainpath"]

    img_dir = os.path.join(root_dir, "imgs")
    mask_dir = os.path.join(root_dir, "masks")

    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Couldn't find 'imgs' or 'masks' in {root_dir}")

    img_names = sorted(os.listdir(img_dir))
    mask_names = sorted(os.listdir(mask_dir))


    logging.info("  - Dataset creation")

    # create tuples (img, mask)
    all_samples = []
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)
        
        if not os.path.exists(mask_path):
            base_name = os.path.splitext(img_name)[0]
            possible_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']
            found = False
            for ext in possible_exts:
                potential_path = os.path.join(mask_dir, base_name + ext)
                if os.path.exists(potential_path):
                    mask_path = potential_path
                    found = True
                    break
            if not found:
                logging.warning(f"Masque non trouvé pour {img_name}, ignoré.")
                continue

        all_samples.append((img_path, mask_path))  
    
    logging.info(f"  - {len(all_samples)} pairs found.")

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

    

    num_classes = 1
    input_size = (3, resize, resize)

    return train_loader, valid_loader, input_size, num_classes
