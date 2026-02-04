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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
            augmented = self.transform(image=image, mask=mask, hm_metadata=[], fda_metadata=[])
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask


def show_image(X):
    num_c = X.shape[0]
    plt.figure()
    plt.imshow(X[0] if num_c == 1 else X.permute(1, 2, 0))
    plt.show()

def load_reference_images(folder_path, limit=50, size=256):
    import glob
    import random
    image_paths = glob.glob(f"{folder_path}/*.jpg")
    if len(image_paths) > limit:
        image_paths = random.sample(image_paths, limit)
    
    refs = []
    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))
        refs.append(img)
    return refs

def identity_fn(x):
    """Fonction qui ne fait rien, remplace lambda x: x pour que Windows soit content."""
    return x

def get_train_transforms(data_config, aug_type="al", ref_images=None):
    resize = data_config["resize"]

    if aug_type == "al":

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

    elif aug_type == "basic":
        return A.Compose([
            A.Resize(width=resize, height=resize),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, p=0.7, border_mode=cv2.BORDER_CONSTANT), 
            A.ToGray(p=1.0),
        
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
            ToTensorV2()
        ])
    
    elif aug_type == "tl_basic":
        return A.Compose([
            A.Resize(width=resize, height=resize),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, p=0.7, border_mode=cv2.BORDER_CONSTANT), 

            A.ToGray(p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
            ToTensorV2()
        ])

    elif aug_type == "tl_occlusion":
        return A.Compose([
            A.Resize(width=resize, height=resize),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, p=0.7, border_mode=cv2.BORDER_CONSTANT), 

            A.CoarseDropout(
                num_holes_range=[1, 3],
                hole_width_range=[int(resize * 0.1),  int(resize * 0.3)], 
                hole_height_range=[int(resize * 0.1),  int(resize * 0.3)],            
                p=0.3
            ),

            A.ToGray(p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
            ToTensorV2()
        ])
    
    elif aug_type == "tl_occlusion_affine":
        return A.Compose([
            A.Resize(width=resize, height=resize),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),

            A.Affine(
                scale=(0.8, 1.2),      # Zoom in/out by 80-120%
                rotate=(-180, 180),      # Rotate by -15 to +15 degrees
                translate_percent=(0, 0.1), # translate by 0-10%
                shear=(-3, 3),          # shear by -10 to +10 degrees
                p=0.8
            ),

            A.CoarseDropout(
                num_holes_range=[1, 3],
                hole_width_range=[int(resize * 0.1),  int(resize * 0.3)], 
                hole_height_range=[int(resize * 0.1),  int(resize * 0.3)],            
                p=0.3
            ),

            A.ToGray(p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
            ToTensorV2()
        ])
    
    elif aug_type == "tl_occlusion_affine_domain":
        return A.Compose([
            A.Resize(width=resize, height=resize),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),

            A.Affine(
                scale=(0.8, 1.2),      # Zoom in/out by 80-120%
                rotate=(-180, 180),      # Rotate by -15 to +15 degrees
                translate_percent=(0, 0.1), # translate by 0-10%
                shear=(-3, 3),          # shear by -10 to +10 degrees
                p=0.8
            ),

            A.GridDistortion(num_steps=5,
                             distort_limit=[-0.25, 0.25]),
            A.RandomGamma(gamma_limit=[30, 100], p=0.5),
            A.ISONoise(color_shift=[0.01, 0.05],
                       intensity=[0.1, 0.5], p=0.3),
            A.OneOf([
                A.MotionBlur(blur_limit=[5, 15],
                         angle_range=[0, 360],
                         direction_range=[-1, 1], p=0.3),
                A.MedianBlur(blur_limit=[3, 7], p=0.3),
            ]),
            
            
            A.Downscale(scale_range=[0.5, 0.9], p=0.1),

            A.CoarseDropout(
                num_holes_range=[1, 3],
                hole_width_range=[int(resize * 0.1),  int(resize * 0.3)], 
                hole_height_range=[int(resize * 0.1),  int(resize * 0.3)],            
                p=0.3
            ),

            A.ToGray(p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
            ToTensorV2()
        ])
    
    elif aug_type == "tl_occlusion_affine_domain_fda":
        
        ######
        # Import images from goal dataset
        import glob
        folder_path = "./active_learning/pool_unlabelled"
        limit=50

        image_paths = glob.glob(f"{folder_path}/*.jpg")
        if len(image_paths) > limit:
            image_paths = random.sample(image_paths, limit)
        
        refs = []
        for p in image_paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE) 
            img = cv2.resize(img, (resize, resize))
            refs.append(img)

        ### Make transformations
        return A.Compose([
            A.Resize(width=resize, height=resize),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),

            A.Affine(
                scale=(0.8, 1.2),      # Zoom in/out by 80-120%
                rotate=(-180, 180),      # Rotate by -15 to +15 degrees
                translate_percent=(0, 0.1), # translate by 0-10%
                shear=(-3, 3),          # shear by -10 to +10 degrees
                p=0.8
            ),

            A.GridDistortion(num_steps=5,
                             distort_limit=[-0.25, 0.25], p=0.3),

            A.OneOf([
                A.HistogramMatching(
                    reference_images=ref_images,
                    read_fn=identity_fn,   
                    blend_ratio=(0.2, 0.5),
                    p=1.0
                ),
                A.FDA(
                    reference_images=ref_images,
                    beta_limit=0.01,
                    read_fn=identity_fn,      
                    p=1.0
                ),
            ], p=0.6),

            A.RandomGamma(gamma_limit=[30, 100], p=0.5),
            A.ISONoise(color_shift=[0.01, 0.05],
                       intensity=[0.1, 0.5], p=0.3),
            A.OneOf([
                A.MotionBlur(blur_limit=[5, 15],
                         angle_range=[0, 360],
                         direction_range=[-1, 1], p=0.3),
                A.MedianBlur(blur_limit=[3, 7], p=0.3),
            ]),
            
            
            A.Downscale(scale_range=[0.5, 0.9], p=0.1),

            A.CoarseDropout(
                num_holes_range=[1, 3],
                hole_width_range=[int(resize * 0.1),  int(resize * 0.3)], 
                hole_height_range=[int(resize * 0.1),  int(resize * 0.3)],            
                p=0.3
            ),

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

def collect_samples_from_dir(root_dir):
    img_dir = os.path.join(root_dir, "imgs")
    mask_dir = os.path.join(root_dir, "masks") # Ou "masks" selon ton arborescence active learning
    
    # Support pour le dossier Active Learning qui s'appelle souvent pool_labelled/images et pool_labelled/masks
    if not os.path.exists(img_dir):
        img_dir = os.path.join(root_dir, "images")
    
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        logging.warning(f"Ignored folder (couldn't find imgs/masks) : {root_dir}")
        return []

    img_names = sorted(os.listdir(img_dir))
    samples = []

    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        
        base_name = os.path.splitext(img_name)[0]
        possible_names = [img_name, base_name + ".png", base_name + ".jpg"]
        
        mask_path = None
        for pname in possible_names:
            p = os.path.join(mask_dir, pname)
            if os.path.exists(p):
                mask_path = p
                break
        
        if mask_path:
            samples.append((img_path, mask_path))
        else:
            # logging.debug(f"Masque manquant pour {img_name}")
            pass
            
    return samples


def get_dataloaders(data_config, use_cuda):
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]
    resize = data_config["resize"]
    root_dir = data_config["trainpath"]
    active_learning = data_config.get("active_learning", False)
    tl_phase = data_config.get("transfer_learning_phase", None)
    augmentations = data_config.get("augmentations", "basic")

    if tl_phase is not None and tl_phase == "gridsearch":
        logging.info("- Dataloaders : Transfer Learning (GridSearch)")

        train_path = data_config["trainpath"]
        goal_path = data_config["goalpath"]

        train_samples = collect_samples_from_dir(train_path)
        valid_samples = collect_samples_from_dir(goal_path)

        logging.info(f"  - Train samples: {len(train_samples)} (from external dataset)")
        logging.info(f"  - Validation samples: {len(valid_samples)} (from goal dataset)")
        logging.info(f"  - Using transformations: {augmentations}")
        
        train_ds = PelgasDataset(
            samples=train_samples, 
            transform=get_train_transforms(data_config, aug_type=augmentations)
        )

        valid_ds = PelgasDataset(
            samples=valid_samples, 
            transform=get_valid_transforms(data_config)
        )

        # 4. Loaders
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_cuda,
            drop_last=True 
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=use_cuda,
        )

        num_classes = 1
        input_size = (3, resize, resize)

        return train_loader, valid_loader, input_size, num_classes

    if not active_learning:
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
            transform=get_train_transforms(data_config, ref_images=load_reference_images(folder_path="./active_learning/pool_unlabelled"))
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

    else:
        # Chemins
        original_data_path = data_config["trainpath"]
        active_learning_path = data_config.get("al_path", "./active_learning/pool_labelled")
        
        # Facteur de poids pour les nouvelles données (ex: 10 signifie qu'une image AL 
        # a 10x plus de chances d'être piochée qu'une image normale)
        al_weight_factor = data_config.get("al_weight", 10.0) 

        logging.info("--- Data Loading ---")

        # 1. Récupération des données ORIGINALES
        original_samples = collect_samples_from_dir(original_data_path)
        logging.info(f"  - Original dataset: {len(original_samples)} samples found.")

        # 2. Récupération des données ACTIVE LEARNING
        al_samples = collect_samples_from_dir(active_learning_path)
        logging.info(f"  - Active Learning dataset: {len(al_samples)} samples found.")

        # 3. Split Train/Val sur les données ORIGINALES uniquement
        # On veut que le set de validation reste stable pour comparer les performances
        val_stratified_path = data_config.get("val_path", r".\active_learning\validation_data_stratified")
        valid_samples = collect_samples_from_dir(val_stratified_path)

        if len(valid_samples) > 0:
            logging.info(f"  - Using Stratified Validation Set: {len(valid_samples)} samples.")
            # Si on a un set de validation externe, on utilise TOUT le dataset original pour le train
            train_samples_orig = original_samples
        else:
            # FALLBACK : Si le dossier est vide ou introuvable, on garde l'ancien comportement (split)
            logging.warning("  - Stratified Val Set not found/empty. Falling back to random split.")
            random.seed(42) 
            random.shuffle(original_samples)
            num_valid = int(valid_ratio * len(original_samples))
            train_samples_orig = original_samples[num_valid:]
            valid_samples = original_samples[:num_valid]


        # 4. Construction du Training Set final
        # On ajoute TOUTES les données Active Learning au Train (ce sont des cas durs)
        full_train_samples = train_samples_orig + al_samples
        
        # 5. Calcul des poids pour le Sampler
        # Poids 1.0 pour les données originales, Poids X pour les données AL
        weights_orig = [1.0] * len(train_samples_orig)
        weights_al = [al_weight_factor] * len(al_samples)
        
        sample_weights = weights_orig + weights_al
        
        # Création du Sampler
        # num_samples : on peut définir combien d'images on tire par époque. 
        # Ici, on garde la taille du dataset total.
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(full_train_samples),
            replacement=True
        )

        # 6. Création des Datasets
        train_ds = PelgasDataset(
            samples=full_train_samples, 
            transform=get_train_transforms(data_config, augmentations=augmentations)
        )

        valid_ds = PelgasDataset(
            samples=valid_samples, 
            transform=get_valid_transforms(data_config)
        )

        # 7. Création des Loaders
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,      # <--- IMPORTANT : On utilise le sampler
            shuffle=False,        # <--- IMPORTANT : shuffle doit être False avec un sampler
            num_workers=num_workers,
            pin_memory=use_cuda,
            drop_last=True
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_cuda,
        )

        logging.info(f"  - Final Train size: {len(train_ds)} (Orig: {len(train_samples_orig)} + AL: {len(al_samples)})")
        logging.info(f"  - Valid size: {len(valid_ds)}")
        if len(al_samples) > 0:
            logging.info(f"  - AL Data Weighting: x{al_weight_factor}")

        num_classes = 1
        input_size = (3, resize, resize)

        return train_loader, valid_loader, input_size, num_classes