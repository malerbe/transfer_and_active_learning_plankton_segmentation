import os
import glob
import random
import shutil
import cv2
import torch
import numpy as np
import napari
from torchvision import transforms
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_DATASET_ROOT = r".\data\original_dataset\train"  
VAL_DESTINATION = r".\active_learning\validation_data_stratified"   
SAMPLES_PER_CLASS = 2                              

USE_MODEL_ASSIST = True
MODEL_PATH = r".\logs\DeepLabV3Plus_6\best_model.pt"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. FONCTIONS UTILITAIRES ---

def get_stratified_samples(root_dir, n_per_class, dest_dir):
    """
    Récupère n images au hasard dans chaque sous-dossier,
    en excluant celles qui existent déjà dans dest_dir/imgs.
    """
    classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    selected_files = []

    # 1. Lister les fichiers déjà annotés pour ne pas les refaire
    dest_imgs_path = os.path.join(dest_dir, "imgs")
    already_done = set()
    if os.path.exists(dest_imgs_path):
        already_done = set(os.listdir(dest_imgs_path))
    
    print(f"Classes trouvées : {len(classes)}")
    print(f"Images déjà annotées détectées : {len(already_done)}")

    for cls in classes:
        cls_path = os.path.join(root_dir, cls)
        # Extensions possibles
        all_images = []
        for ext in ["*.jpg", "*.png", "*.jpeg", "*.tif"]:
            all_images.extend(glob.glob(os.path.join(cls_path, ext)))

        if len(all_images) == 0:
            # print(f"Warning: Pas d'images source dans {cls}")
            continue

        # 2. Filtrer les images déjà faites
        # La logique de nommage est : "{cls}_{filename}" (voir annotate_and_move)
        available_images = []
        for img_path in all_images:
            filename = os.path.basename(img_path)
            expected_name = f"{cls}_{filename}" # Le nom qu'aura le fichier une fois annoté
            
            if expected_name not in already_done:
                available_images.append(img_path)
        
        # S'il ne reste rien à faire pour cette classe
        if len(available_images) == 0:
            print(f"   -> Classe '{cls}' déjà complète (ou vide). On passe.")
            continue

        # 3. Tirage au sort parmi celles RESTANTES
        # Si on demande 2 images mais qu'il n'en reste qu'1 seule non faite, on prend tout ce qu'il reste.
        n = min(len(available_images), n_per_class)
        picked = random.sample(available_images, n)
        
        # On garde le chemin et le nom de la classe pour le renommage futur
        for p in picked:
            selected_files.append({"path": p, "class": cls})

    random.shuffle(selected_files)
    return selected_files

def load_model_inference(path, device):
    # Adapter selon tes imports exacts
    from torchtmpl.models.seg_models import DeepLabV3Plus
    model = DeepLabV3Plus({"backbone": "resnet34", "pretrained": False}, (3, IMG_SIZE, IMG_SIZE), 1)
    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

# --- 2. ANNOTATION ---

def annotate_and_move(samples, dest_root, model=None):
    img_dest_dir = os.path.join(dest_root, "imgs")
    mask_dest_dir = os.path.join(dest_root, "masks")
    os.makedirs(img_dest_dir, exist_ok=True)
    os.makedirs(mask_dest_dir, exist_ok=True)

    # Viewer Napari
    viewer = napari.Viewer(title="Validation Set Annotation")

    # Transformation pour le modèle (Normalisation)
    tfm_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    state = {'index': 0}

    def load_current():
        if state['index'] >= len(samples):
            print(">>> Toutes les images sélectionnées ont été traitées !")
            viewer.close()
            return

        sample = samples[state['index']]
        original_path = sample['path']
        cls_name = sample['class']

        # Nom unique pour le fichier final : Classe_NomFichier.jpg
        filename = os.path.basename(original_path)
        new_filename = f"{cls_name}_{filename}"
        sample['new_filename'] = new_filename 

        # Lecture Image
        image_bgr = cv2.imread(original_path)
        if image_bgr is None:
            print(f"Erreur lecture image: {original_path}")
            state['index'] += 1
            load_current()
            return
            
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Prédiction (Pré-Remplissage)
        initial_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

        if model:
            # Pour le modèle, on doit resize à 224x224
            input_img = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
            input_tensor = tfm_normalize(input_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                preds = model(input_tensor)
                probs = torch.sigmoid(preds)
                pred_mask = (probs > 0.5).float().cpu().numpy()[0, 0]

            # On remet le masque à la taille originale de l'image
            initial_mask = cv2.resize(pred_mask, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            initial_mask = initial_mask.astype(int)

        viewer.layers.clear()
        viewer.add_image(image_rgb, name='Image', rgb=True)
        
        # Correction add_labels sans num_colors
        lbl_layer = viewer.add_labels(initial_mask, name='Mask')
        lbl_layer.editable = True
        # On peut pré-sélectionner le label 1 pour dessiner direct
        lbl_layer.selected_label = 1

        viewer.text_overlay.visible = True
        viewer.text_overlay.text = f"Image {state['index']+1}/{len(samples)}\nClasse : {cls_name}\n(S) pour Sauvegarder et Suivant"

    @viewer.bind_key('s')
    def save_and_next(viewer):
        sample = samples[state['index']]

        # Récupérer le masque dessiné
        try:
            mask_layer = viewer.layers['Mask']
            final_mask = mask_layer.data.astype(np.uint8)
        except KeyError:
            print("Erreur : Pas de calque 'Mask' trouvé.")
            return

        # Chemins de destination
        new_name = sample['new_filename']
        base_name = os.path.splitext(new_name)[0]

        dest_img_path = os.path.join(img_dest_dir, new_name)
        dest_mask_path = os.path.join(mask_dest_dir, base_name + ".png") 

        # 1. Sauvegarde du masque (x255 pour visibilité)
        success_mask = cv2.imwrite(dest_mask_path, final_mask * 255)

        # 2. Copie de l'image originale
        try:
            shutil.copy(sample['path'], dest_img_path)
            success_img = True
        except Exception as e:
            print(f"Erreur copie image : {e}")
            success_img = False

        if success_mask and success_img:
            print(f"Sauvegardé : {new_name}")
            state['index'] += 1
            load_current()
        else:
            print("ERREUR lors de la sauvegarde !")

    load_current()
    napari.run()

# --- MAIN ---

if __name__ == "__main__":
    # 1. Sélection des images
    print("--- Sélection des échantillons ---")
    # On passe VAL_DESTINATION pour vérification
    samples = get_stratified_samples(SOURCE_DATASET_ROOT, SAMPLES_PER_CLASS, VAL_DESTINATION)
    
    print(f"Total images restantes à annoter : {len(samples)}")

    if len(samples) == 0:
        print("Rien à faire, tout est déjà annoté ou dossiers vides.")
        exit()

    # 2. Chargement Modèle (Optionnel)
    model = None
    if USE_MODEL_ASSIST:
        try:
            print("Chargement du modèle pour assistance...")
            model = load_model_inference(MODEL_PATH, DEVICE)
        except Exception as e:
            print(f"Impossible de charger le modèle ({e}). Passage en manuel pur.")

    # 3. Lancement
    print("--- Lancement Napari ---")
    annotate_and_move(samples, VAL_DESTINATION, model)
