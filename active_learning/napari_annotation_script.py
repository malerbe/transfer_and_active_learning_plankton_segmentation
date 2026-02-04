#############################
# author: Gemini Pro 3
#############################
import napari
import shutil
import torch
import numpy as np
import cv2
import os
from PIL import Image
from torchvision import transforms
import cv2

def annotate_batch(image_paths, model, save_dir, device, input_size=(224, 224)):
    """
    Ouvre une session Napari pour une liste d'images.
    """
    # Vérification dossier sauvegarde
    if not os.path.exists(save_dir):
        print(f"Création du dossier de sauvegarde : {save_dir}")
        os.makedirs(save_dir, exist_ok=True)

    # État interne
    state = {
        'index': 0,
        'current_path': None,
        'original_size': None
    }

    # Initialiser le viewer
    viewer = napari.Viewer(title="Active Learning Annotation")

    # --- Préparation des transformations (IDENTIQUES à l'entraînement/sélection) ---
    # C'est crucial : sans ça, le modèle voit l'image "déformée" niveau couleurs
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def load_current_image():
        if state['index'] >= len(image_paths):
            print(">>> Toutes les images ont été traitées ! Fermeture.")
            viewer.close()
            return

        image_path = image_paths[state['index']]
        state['current_path'] = image_path
        
        print(f"[{state['index']+1}/{len(image_paths)}] Chargement : {os.path.basename(image_path)}")

        # 1. Charger l'image originale (RGB)
        img_pil = Image.open(image_path).convert('RGB')
        img_np = np.array(img_pil) # (H, W, 3) en uint8 (0-255)
        h_orig, w_orig = img_np.shape[:2]
        state['original_size'] = (h_orig, w_orig)

        # 2. Prétraitement correct pour le modèle
        img_resized = img_pil.resize(input_size, Image.BILINEAR)
        # On applique la normalisation Mean/Std ici
        input_tensor = transform_pipeline(img_resized).unsqueeze(0).to(device)

        # 3. Prédiction
        with torch.no_grad():
            preds = model(input_tensor)
            probs = torch.sigmoid(preds)
            mask_pred_small = (probs > 0.5).squeeze().cpu().numpy().astype(np.uint8)

        # 4. Resize du masque à la taille d'origine
        mask_pred_full = cv2.resize(mask_pred_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        # 5. Mise à jour de Napari
        try:
            # On met à jour les données existantes
            viewer.layers['Image'].data = img_np
            viewer.layers['Masque'].data = mask_pred_full
        except KeyError:
            # Création des calques (rgb=True force le bon affichage des couleurs)
            viewer.add_image(img_np, name='Image', rgb=True) 
            viewer.add_labels(mask_pred_full, name='Masque', opacity=0.4)
            
            # Réglage caméra (optionnel, pour centrer si besoin)
            viewer.reset_view()

    @viewer.bind_key('s')
    def save_and_next(viewer):
        # 1. Récupérer le masque
        final_mask = viewer.layers['Masque'].data.astype(np.uint8)

        # 2. Chemin
        filename = os.path.basename(state['current_path'])
        save_name = os.path.splitext(filename)[0] + ".png"
        save_path = os.path.join(save_dir, save_name)

        # 3. Sauvegarde sécurisée
        # cv2.imwrite renvoie True si succès, False sinon
        success = cv2.imwrite(save_path, final_mask * 255)
        
        if success:
            print(f" -> OK : Masque sauvegardé sous {save_name}")
            # 4. Suivante
            state['index'] += 1
            load_current_image()
        else:
            print(f" -> ERREUR CRITIQUE : Impossible de sauvegarder dans {save_path}")
            print(" -> Vérifiez que le dossier existe et que les permissions sont bonnes.")

    # Lancer la première
    load_current_image()
    napari.run()


def move_files_to_labeled_pool(annotated_images_list, mask_dir, labeled_image_dir):
    """
    Déplace les images annotées du pool 'unlabeled' vers 'pool_labeled/images'
    UNIQUEMENT si le masque correspondant a bien été sauvegardé.
    """
    # Créer le dossier de destination s'il n'existe pas
    os.makedirs(labeled_image_dir, exist_ok=True)
    
    count = 0
    print("\n--- Organisation des fichiers ---")

    for img_path in annotated_images_list:
        filename = os.path.basename(img_path)
        
        # Le nom du masque attendu (ton annotate_batch force l'extension .png)
        mask_name = os.path.splitext(filename)[0] + ".png"
        mask_path = os.path.join(mask_dir, mask_name)

        # Vérification : Est-ce que tu as bien appuyé sur 'S' pour cette image ?
        if os.path.exists(mask_path):
            dst_path = os.path.join(labeled_image_dir, filename)
            try:
                shutil.move(img_path, dst_path)
                # print(f"Déplacé : {filename}") # Decommenter pour verbeux
                count += 1
            except Exception as e:
                print(f"ERREUR lors du déplacement de {filename}: {e}")
        else:
            print(f"IGNORÉ : Pas de masque trouvé pour {filename} (Reste dans Unlabeled)")

    print(f">>> {count} images déplacées avec succès vers {labeled_image_dir}")
#############################################