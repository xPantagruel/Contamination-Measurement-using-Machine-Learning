# @file AugmentateDataset.py
# @brief file to create augmented images for our finetuning process
# @author Matěj Macek (xmacek27@stud.fit.vutbr.cz)
# @date 4.5.2024

import cv2
import numpy as np
import albumentations as A
import os
from multiprocessing import Pool

# fill in these file destinations, file with masks and images you want to augmentate and also files where you want to store new augmented images and masks
image_dir = r''
output_image_dir =r''
mask_dir = r''
output_mask_dir = r''

# The augmentation pipeline
augmentation_pipeline = A.Compose([
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=1),
        A.Blur(blur_limit=3, p=1),
        A.MedianBlur(blur_limit=3, p=1),
    ], p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.RandomScale(scale_limit=0.1, p=0.7),
    A.ShiftScaleRotate(
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=0,
        p=1.0,
        border_mode=cv2.BORDER_REFLECT  # Changed to BORDER_REFLECT to avoid black borders
    ),
    A.Resize(320, 480)  # Resize all images 
], additional_targets={'mask': 'image'})  # Apply the same transformation to the mask

def augment_image_and_mask(args):
    image_path, mask_path, output_image_dir, output_mask_dir, num_augmented_images, i = args
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        print(f"Could not read the image or mask {image_path}, {mask_path}. Skipping...")
        return
    
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=-1)

    image_name, ext = os.path.splitext(os.path.basename(image_path))

    augmented = augmentation_pipeline(image=image, mask=mask)
    augmented_image = augmented['image']
    augmented_mask = augmented['mask']

    augmented_image_path = f"{output_image_dir}/{image_name}_aug_{i+1}{ext}"
    augmented_mask_path = f"{output_mask_dir}/{image_name}_aug_{i+1}{ext}"
    
    cv2.imwrite(augmented_image_path, augmented_image)
    cv2.imwrite(augmented_mask_path, augmented_mask)
    print(f"Image and mask saved to {augmented_image_path} and {augmented_mask_path}")

def augment_images_and_masks_in_directory(args):
    image_dir, mask_dir, output_image_dir, output_mask_dir, num_augmented_images = args
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    params = [(os.path.join(image_dir, f), os.path.join(mask_dir, f), output_image_dir, output_mask_dir, num_augmented_images, i) for f in files for i in range(num_augmented_images)]

    with Pool() as pool:
        pool.map(augment_image_and_mask, params)
        
if __name__ == '__main__':
    augment_images_and_masks_in_directory((image_dir, mask_dir, output_image_dir, output_mask_dir, 10))
