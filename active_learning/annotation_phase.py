# usual imports
import glob
import os
from tqdm import tqdm
import shutil
import torch
import random

# External imports
from torchtmpl.models.seg_models import *
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Local imports
from napari_annotation_script import *

#######################
# Configuration
#######################
SAMPLES_PER_ITERATION = 10
UNLABELLED_POOL = "./active_learning/pool_unlabelled"
LABELLED_POOL = "./active_learning/pool_labelled"

# Model which will be used both for uncertainty sampling and to 
# provide an initialization mask for the manual annotation
PRE_TRAINED_MODEL = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\transfer_and_active_learning_plankton_segmentation\logs\Unet_21\best_model.pt"

# Model configuration
MODEL_ARCH = "Unet"      
BACKBONE = "resnet34"
PRETRAINED = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512
input_size = (3, IMG_SIZE, IMG_SIZE)
num_classes = 1

MODEL_MAP = {
    "Unet": Unet,
    "DeepLabV3": DeepLabV3,
    "DeepLabV3Plus": DeepLabV3Plus,
    "UnetPlusPlus": UnetPlusPlus
}

#######################
# Utilitary functions
#######################
def load_model(model_arch, path, backbone, device, pretrained=False):
    cfg = {"backbone": backbone, "pretrained": pretrained} 

    model_class = MODEL_MAP[model_arch]
    model = model_class(cfg, input_size, num_classes)
    
    # load weights
    try:
        checkpoint = torch.load(path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded {model_arch} successfully.")
    except Exception as e:
        print(f"Error while loading weights : {e}")
        try:
            model.load_state_dict(checkpoint, strict=False)
        except:
            pass

    model.to(device)
    model.eval()
    return model

def select_uncertain_samples(model, unlabeled_dir, n_samples):
    # get all unlabelled images
    all_images = glob.glob(os.path.join(unlabeled_dir, "*.jpg"))

    scores = []
    model.eval()

    with torch.no_grad():
        for img_path in tqdm(all_images):
            # preprocess
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.ToGray(p=1.0),
            ])

            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)

            preds = model(input_tensor) # Logits
            probs = torch.sigmoid(preds) # 0 à 1

            uncertainty_map = 1 - (2 * torch.abs(probs - 0.5)) 

            avg_uncertainty = torch.mean(uncertainty_map).item()
            scores.append((img_path, avg_uncertainty))

    scores.sort(key=lambda x: x[1], reverse=True)

    return [x[0] for x in scores[:n_samples]]

def select_uncertain_samples_stochastic(model, unlabeled_dir, n_samples, n_batches):
    """Uncertainty sampling inspired by the paper

    ACTIVE LEARNING FOR MEDICAL IMAGE SEGMENTATION WITH STOCHASTIC BATCHES

    https://arxiv.org/abs/2301.07670

    n_samples = how many samples to extract
    n_batches = how many batches of n_samples to test
    """

    # get images:
    all_images = glob.glob(os.path.join(unlabeled_dir, "*.jpg"))
    
    # generate n_batched of n_samples images randomly
    candidate_batches = []
    unique_indices = []
    for _ in range(n_batches):
        batch_indices = random.sample(range(len(all_images)), n_samples)
        candidate_batches.append(batch_indices)
        unique_indices.append(batch_indices)

    # flatten unique indices to make it a 1-dimensional list
    all_indices_flat = [idx for batch in candidate_batches for idx in batch]
    unique_indices = list(set(all_indices_flat)) 

    # Note: because the models inference are really fast and the amount of available images
    # huge, we don't check if some images are present in multiple batches

    print("CAUTION: Please make sure that the transformations used for uncertainty computation\
 are the same as the one used the validation.")
    model.eval()
    base_transform = A.Compose([
        A.Resize(width=IMG_SIZE, height=IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    noise_transform = A.Compose([
        A.GaussNoise(var_limit=(5.0, 15.0), p=1.0),
        A.Resize(width=IMG_SIZE, height=IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    print(f"Computing uncertainty for {n_batches * n_samples} images...")
    uncertainties = {}
    with torch.no_grad():
        for index in tqdm(unique_indices):
            img_path = all_images[index]

            original_image = cv2.imread(str(img_path))
            # print(img_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            # --- TTA ---
            preds_stack = []

            aug = base_transform(image=original_image)
            pred = torch.sigmoid(model(aug['image'].unsqueeze(0).to(DEVICE)))
            preds_stack.append(pred)

            ## horizontal flip
            image_hf = cv2.flip(original_image, 1)
            aug = base_transform(image=image_hf)
            pred = torch.sigmoid(model(aug['image'].unsqueeze(0).to(DEVICE)))
            # inverse transformation
            pred_inv = torch.flip(pred, dims=[3]) 
            preds_stack.append(pred_inv)

            ## vertical flip
            image_vf = cv2.flip(original_image, 0)
            aug = base_transform(image=image_vf)
            pred = torch.sigmoid(model(aug['image'].unsqueeze(0).to(DEVICE)))
            # inverse transformation
            pred_inv = torch.flip(pred, dims=[2])
            preds_stack.append(pred_inv)

            # rotate 90
            image_r90 = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)
            aug = base_transform(image=image_r90)
            pred = torch.sigmoid(model(aug['image'].unsqueeze(0).to(DEVICE)))
            # inverse
            pred_inv = torch.rot90(pred, k=-1, dims=[2, 3])
            preds_stack.append(pred_inv)

            # gaussian noise
            aug = noise_transform(image=original_image)
            pred = torch.sigmoid(model(aug['image'].unsqueeze(0).to(DEVICE)))
            preds_stack.append(pred)

            # --- End of TTA ---
            # Compute uncertainty:
            stack = torch.cat(preds_stack, dim=0)
            variance_map = torch.var(stack, dim=0) 

            avg_uncertainty = torch.mean(variance_map).item()
            uncertainties[index] = avg_uncertainty

    print("Computing batches uncertainty...")
    best_batch_score = -1
    best_batch_indices = []
    for batch in candidate_batches:
        scores = [uncertainties[index] for index in batch]
        mean_score = np.mean(scores)

        if mean_score > best_batch_score:
            best_batch_score = mean_score
            best_batch_indices = batch

    selected_paths = [all_images[idx] for idx in best_batch_indices]
    
    print(f"Batch selected with uncertainty {best_batch_score}")
    return selected_paths


#######################
# Script
#######################
def main():
    current_model = PRE_TRAINED_MODEL

    labeled_img_dir = os.path.join(LABELLED_POOL, "images")
    labeled_mask_dir = os.path.join(LABELLED_POOL, "masks")

    cycle = 0
    while True:
        cycle += 1
        print("=====================================")
        print(f"Starting cycle {cycle}")

        # load current model
        print(f"Loading model")
        model = load_model(model_arch=MODEL_ARCH,
                           path=current_model,
                           backbone=BACKBONE,
                           device=DEVICE,
                           pretrained=PRETRAINED)
        
        print(f"Inference to select most uncertain samples")
        uncertain_samples = select_uncertain_samples_stochastic(model, UNLABELLED_POOL, SAMPLES_PER_ITERATION, n_batches=50)

        print("Opening Napari for annotation")
        annotate_batch(uncertain_samples, model, save_dir=labeled_mask_dir, device=DEVICE)

        print("moving labelled images from unlabelled pool to labelled pool")
        move_files_to_labeled_pool(uncertain_samples, labeled_mask_dir, labeled_img_dir)

        while True:
            user_input = input("\n PATH to new model : ").strip()
            clean_path = user_input.replace('"', '').replace("'", "")
            if os.path.isfile(clean_path):
                break
        print(f"Fin du cycle {cycle}.")
        
main()

        

        

