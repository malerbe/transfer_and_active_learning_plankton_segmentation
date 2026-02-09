import os
import glob
import cv2
import numpy as np
import napari
import random

# --- CONFIGURATION ---
# Doit pointer vers le même dossier que VAL_DESTINATION du script précédent
VAL_DATASET_ROOT = r".\active_learning\pool_labelled"

IMG_DIR = os.path.join(VAL_DATASET_ROOT, "images")
MASK_DIR = os.path.join(VAL_DATASET_ROOT, "masks")

# --- FONCTIONS ---

def load_dataset_files(img_dir, mask_dir):
    """
    Récupère la liste des paires (image, masque) triée par nom.
    """
    # Extensions d'images supportées
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.bmp"]
    img_paths = []
    for ext in extensions:
        img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
    
    # Tri alphabétique pour garder l'ordre constant entre les sessions
    random.shuffle(img_paths)
    
    dataset = []
    for img_p in img_paths:
        filename = os.path.basename(img_p)
        basename = os.path.splitext(filename)[0]
        
        # On déduit le chemin du masque (toujours en .png dans le script précédent)
        mask_p = os.path.join(mask_dir, basename + ".png")
        
        if os.path.exists(mask_p):
            dataset.append({
                "img_path": img_p,
                "mask_path": mask_p,
                "name": filename
            })
        else:
            print(f"[ATTENTION] Masque manquant pour : {filename} (ignoré)")
            
    return dataset

def review_process(dataset):
    if not dataset:
        print("Aucune paire Image/Masque trouvée. Vérifie les chemins.")
        return

    viewer = napari.Viewer(title="Review Mode - Validation Set")
    
    # État global pour gérer l'index courant
    state = {'index': 0}

    def load_current_sample():
        """Charge l'image et le masque à l'index actuel"""
        if state['index'] < 0:
            state['index'] = 0
        if state['index'] >= len(dataset):
            print(">>> Fin de la liste ! Tu peux fermer Napari.")
            viewer.text_overlay.text = "FIN DE LA LISTE"
            return

        sample = dataset[state['index']]
        
        # 1. Lecture Image
        image_bgr = cv2.imread(sample['img_path'])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. Lecture Masque
        # Le masque est sauvegardé en 0-255 (noir et blanc visible)
        # Napari préfère les labels en entiers (0, 1, 2...)
        mask_img = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
        
        # Conversion 255 -> 1 pour l'édition (Binarisation propre)
        mask_label = (mask_img > 127).astype(int)

        # 3. Mise à jour Viewer
        viewer.layers.clear()
        
        viewer.add_image(image_rgb, name='Image')
        
        lbl_layer = viewer.add_labels(mask_label, name='Mask')
        lbl_layer.editable = True
        lbl_layer.selected_label = 1 # Sélectionne le "pinceau" blanc par défaut

        # Info textuelle
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = (
            f"Image {state['index'] + 1} / {len(dataset)}\n"
            f"Fichier : {sample['name']}\n\n"
            f"[S] Sauvegarder et Suivant\n"
            f"[B] Retour (Précédent)\n"
            f"[D] Supprimer ce masque (Remet à zéro)"
        )

    # --- KEY BINDINGS ---

    @viewer.bind_key('s')
    def save_and_next(viewer):
        """Sauvegarde les modifs et passe au suivant"""
        if state['index'] >= len(dataset):
            return

        try:
            mask_layer = viewer.layers['Mask']
            # On récupère les données (0 ou 1)
            edited_mask = mask_layer.data.astype(np.uint8)
            
            # On remet en 255 pour la sauvegarde disque
            final_mask_disk = edited_mask * 255
            
            current_mask_path = dataset[state['index']]['mask_path']
            
            # Écrasement du fichier
            cv2.imwrite(current_mask_path, final_mask_disk)
            print(f"[OK] Modifié : {os.path.basename(current_mask_path)}")
            
            # Suivant
            state['index'] += 1
            load_current_sample()
            
        except Exception as e:
            print(f"Erreur sauvegarde : {e}")

    @viewer.bind_key('b')
    def previous_image(viewer):
        """Revient en arrière sans sauvegarder"""
        if state['index'] > 0:
            print("<- Retour arrière")
            state['index'] -= 1
            load_current_sample()
        else:
            print("Déjà au début de la liste.")

    @viewer.bind_key('d')
    def clear_mask(viewer):
        """Raccourci pour tout effacer (remplir de 0)"""
        try:
            mask_layer = viewer.layers['Mask']
            mask_layer.data = np.zeros_like(mask_layer.data)
            mask_layer.refresh()
            print("Masque effacé (en mémoire). Appuie sur S pour valider.")
        except:
            pass

    # Lancement initial
    print("Chargement de la première image...")
    load_current_sample()
    napari.run()

if __name__ == "__main__":
    print(f"--- MODE REVIEW ---")
    print(f"Dossier cible : {VAL_DATASET_ROOT}")
    
    data = load_dataset_files(IMG_DIR, MASK_DIR)
    print(f"Images chargées : {len(data)}")
    
    review_process(data)
