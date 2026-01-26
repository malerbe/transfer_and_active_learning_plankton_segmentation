# imports
import os
import cv2
import numpy as np
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import urllib.request
from segment_anything import sam_model_registry, SamPredictor
import random
from skimage import io, color, filters, measure, morphology
import matplotlib.pyplot as plt

# configuration
INPUT_DATASET = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\transfer_and_active_learning_plankton_segmentation\data\pelgas_dataset_temp\101141\individual_images"
OUTPUT_DATASET = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\transfer_and_active_learning_plankton_segmentation\data\segmentation_dataset_cropped"
#SAM_MODEL = "./sam_vit_b_01ec64.pth"
SAM_MODEL = "./sam_vit_h_4b8939.pth"
IMGS_BY_CLASS = 40 # max nbr of images taken in each class to build the dataset
BOTTOM_TOLERANCE = 20
USE_SAM = False

##################################### useful function
def smart_crop_background_deviation(img, margin_ratio=0.10, noise_tolerance=5):
    """Détecte le plancton et retourne l'image, le masque et la bbox."""
    if img.ndim == 3:
        gray = color.rgb2gray(img)
        if gray.max() <= 1.0:
            gray = (gray * 255).astype(np.uint8)
    else:
        gray = img
        if gray.dtype != np.uint8 and gray.max() <= 1.0:
            gray = (gray * 255).astype(np.uint8)
    
    h, w = gray.shape

    try:
        thresh_rough = filters.threshold_otsu(gray)
        mask_rough = gray < thresh_rough
        mask_rough = morphology.dilation(mask_rough, morphology.disk(6))
        lbl_rough = measure.label(mask_rough)
        regions_rough = measure.regionprops(lbl_rough)
        
        scale_top_limit = h 
        found_scale = False
        
        for r in regions_rough:
            minr, minc, maxr, maxc = r.bbox
            if maxr >= h - BOTTOM_TOLERANCE:
                found_scale = True
                if minr < scale_top_limit:
                    scale_top_limit = minr

        if found_scale:
            scale_top_limit = max(0, scale_top_limit - 5)
    except:
        scale_top_limit = h

    bg_sample = gray[0:h//2, :]
    bg_value = np.median(bg_sample)
    threshold_value = bg_value - noise_tolerance
    
    mask_fine = (gray < threshold_value)
    mask_fine[scale_top_limit:h, :] = False
    mask_plankton = morphology.dilation(mask_fine, morphology.disk(2))

    coords = np.argwhere(mask_plankton)
    bbox = None
    
    if coords.size > 0:
        minr, minc = coords.min(axis=0)
        maxr, maxc = coords.max(axis=0)
        
        h_box, w_box = maxr - minr, maxc - minc
        pad_h = max(int(h_box * margin_ratio), 10)
        pad_w = max(int(w_box * margin_ratio), 10)

        safe_minr = max(0, minr - pad_h)
        safe_minc = max(0, minc - pad_w)
        safe_maxr = min(scale_top_limit, maxr + pad_h)
        safe_maxc = min(w, maxc + pad_w)
        
        if safe_maxr > safe_minr and safe_maxc > safe_minc:
            bbox = (safe_minr, safe_minc, safe_maxr, safe_maxc)

    return img, mask_plankton, bbox
if USE_SAM:
    # setup SAM
    # url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

    # if model weights are not downloaded yet...
    if not os.path.exists(SAM_MODEL):
        urllib.request.urlretrieve(url, SAM_MODEL)

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=SAM_MODEL)
    sam.to(device=device)

    predictor = SamPredictor(sam)

## process dataset
# prepare paths and creat folders
input_path = Path(INPUT_DATASET)
output_imgs = Path(OUTPUT_DATASET) / "imgs"
output_masks = Path(OUTPUT_DATASET) / "masks"

output_imgs.mkdir(parents=True, exist_ok=True)
output_masks.mkdir(parents=True, exist_ok=True)

# find classes
classes = [d for d in input_path.iterdir() if d.is_dir()]

print(f"Found {len(classes)} classes")

# process class by class
total = 0
for _class in tqdm(classes):
    class_name = _class.name
    images = list(_class.glob("*.jpg"))

    random.shuffle(images)
    selected_images = images[:IMGS_BY_CLASS]
    for img_path in selected_images:
        # read image
        img_array = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        ## 1. apply 
        # convert to grayscale
        img, mask, bbox = smart_crop_background_deviation(img)

        # plt.imshow(img)
        # plt.show()

        # plt.imshow(mask)
        # plt.show()

        if bbox is None:
            continue

        y_min, x_min, y_max, x_max = bbox
        box_promt = np.array([x_min, y_min, x_max, y_max])

        mask_uint8 = mask.astype(np.uint8)

        dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)

        # take a "central" position of the plankton (guaranted to be within the mask)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)

        if max_val > 0:
            point_coords = np.array([max_loc])
            point_labels = np.array([1])


        ## 2. SAM prediction
        if USE_SAM:
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)

            masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box_promt[None, :],
                    multimask_output=True
                )

            # keep best mask
            best_idx = np.argmax(scores)
            final_mask = masks[best_idx].astype(np.uint8) * 255
        
        # crop image
        img_cropped = img[y_min:y_max, x_min:x_max]

        if USE_SAM:
            mask_cropped = final_mask[y_min:y_max, x_min:x_max]
        else:
            mask_full = mask_uint8 * 255
            mask_cropped = mask_full[y_min:y_max, x_min:x_max]


        # plt.imshow(final_mask)
        # plt.show()
        
        ## 3. Save image and predicted mask
        name = f"{class_name}__{img_path.name}"
        mask_name = name.replace(".jpg", ".png")

        cv2.imwrite(str(output_imgs / name), img_cropped)
        cv2.imwrite(str(output_masks / mask_name), mask_cropped)

        total += 1

    print(f"Processed {total} images in class {class_name}, saved in {OUTPUT_DATASET}")



