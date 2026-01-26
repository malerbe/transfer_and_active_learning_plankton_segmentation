import napari
import torch
import numpy as np
import cv2
import os
from PIL import Image

# --- CONFIGURATION ---
IMAGE_PATH = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\transfer_and_active_learning_plankton_segmentation\data\test_image\test_image\334781155.jpg" # L'image choisie par l'active learning
SAVE_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\transfer_and_active_learning_plankton_segmentation\data\test_image" # Où sauvegarder le masque corrigé
MODEL_PATH = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\transfer_and_active_learning_plankton_segmentation\logs\DeepLabV3Plus_6\best_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Charger le modèle (A adapter selon ta classe)
from torchtmpl.models.seg_models import *
backbone = "resnet34"  
IMG_SIZE = 224 
cfg = {"backbone": backbone, "pretrained": False} 
input_size = (3, IMG_SIZE, IMG_SIZE)
num_classes = 1

model = DeepLabV3Plus(cfg, input_size, num_classes)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(DEVICE).eval()

def annotate_image(image_path, model):
    # Charger image originale
    img_pil = Image.open(image_path).convert('RGB')
    img_np = np.array(img_pil)
    
    # On garde les dimensions originales en mémoire
    original_h, original_w = img_np.shape[:2]

    # --- 1. PRETRAITEMENT ---
    # On redimensionne l'image vers 224x224 pour que le modèle soit content
    # (Attention : cv2.resize attend (Width, Height))
    img_resized = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
    
    input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    # --- 2. PREDICTION ---
    with torch.no_grad():
        preds = model(input_tensor)
        probs = torch.sigmoid(preds)
        # Masque en 224x224
        mask_pred_small = (probs > 0.5).squeeze().cpu().numpy().astype(np.uint8)

    # --- 3. POST-TRAITEMENT ---
    # On remet le masque à la taille originale (1019x1232)
    # INTER_NEAREST est OBLIGATOIRE pour garder des valeurs 0 ou 1 (pas de flou)
    mask_pred_full = cv2.resize(mask_pred_small, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    # --- 4. NAPARI ---
    viewer = napari.Viewer(title="Correction Active Learning")

    # Ajouter l'image de fond (Originale HD)
    viewer.add_image(img_np, name='Image')

    # Ajouter le masque (Redimensionné HD)
    viewer.add_labels(mask_pred_full, name='Masque', opacity=0.5)

    print(f"Annotation de {os.path.basename(image_path)}...")
    print("Outils : P (Pinceau), E (Gomme). Appuie sur 'S' pour sauvegarder et quitter.")

    @viewer.bind_key('s')
    def save_mask(viewer):
        # Récupérer le masque corrigé (C'est un numpy array à la taille originale)
        final_mask = viewer.layers['Masque'].data.astype(np.uint8)

        save_name = os.path.basename(image_path).replace(".jpg", ".png")
        # On remplace l'extension si c'est .jpeg ou autre
        if not save_name.endswith(".png"):
             save_name = os.path.splitext(save_name)[0] + ".png"
             
        save_path = os.path.join(SAVE_DIR, save_name)

        # Sauvegarder (0-255 pour être visible)
        cv2.imwrite(save_path, final_mask * 255)

        print(f"Sauvegardé sous : {save_path}")
        viewer.close()

    napari.run()


# Lancer
annotate_image(IMAGE_PATH, model)
