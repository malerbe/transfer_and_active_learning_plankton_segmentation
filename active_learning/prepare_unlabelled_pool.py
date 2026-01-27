import os
import glob
import random
import shutil

# Configuration:
PATH_TO_UNLABELLED_DATASET = r"./data/original_dataset/train"
POOL_UNLABELLED_PATH = r"./active_learning/pool_unlabelled"
MAX_IMG_BY_CLASS = 300

# Get classes
classes = [_class.replace("\\", "/").split("/")[-1] for _class in glob.glob(PATH_TO_UNLABELLED_DATASET + "/*")]

# Get samples in each class
for _class in classes:
    images_paths = [os.path.join(PATH_TO_UNLABELLED_DATASET, _class, img_name).replace("\\", "/") for img_name in os.listdir(os.path.join(PATH_TO_UNLABELLED_DATASET, _class))]
    random.seed(42)
    random.shuffle(images_paths)
    
    selected_imgs = images_paths[:min(len(images_paths), MAX_IMG_BY_CLASS)]

    for selected_img in selected_imgs:
        class_name = selected_img.split("/")[-2]
        img_name = selected_img.split("/")[-1]
        shutil.copy(selected_img, POOL_UNLABELLED_PATH + f"/{class_name}_{img_name}")

    



