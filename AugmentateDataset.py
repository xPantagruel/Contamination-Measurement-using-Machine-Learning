import cv2
import numpy as np
import albumentations as A
import os

# Define your augmentation pipeline, ensuring it is suitable for both images and masks
augmentation_pipeline = A.Compose([
    A.Rotate(limit=15, p=0.7),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.7),
    A.RandomScale(scale_limit=0.1, p=0.7),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.7),
    A.OneOf([
        A.Blur(blur_limit=3, p=1),
        A.MedianBlur(blur_limit=3, p=1),
    ], p=0.7),
], additional_targets={'mask': 'image'})  # This ensures the same transformation is applied to the mask

def augment_image_and_mask(image_path, mask_path, output_image_dir, output_mask_dir, num_augmented_images=10):
    """
    Read an image and its mask, apply augmentation pipeline, and save augmented images and masks.
    """
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Assuming masks are grayscale
    if image is None or mask is None:
        print(f"Could not read the image or mask {image_path}, {mask_path}. Skipping...")
        return
    # Before applying augmentations, check and adjust the mask shape
    if len(mask.shape) == 2:  # If the mask is grayscale
        mask = np.expand_dims(mask, axis=-1)  # Add a single channel dimension to match the [H, W, C] format

    image_name, ext = os.path.splitext(os.path.basename(image_path))

    for i in range(num_augmented_images):
        augmented = augmentation_pipeline(image=image, mask=mask)
        augmented_image = augmented['image']
        augmented_mask = augmented['mask']

        augmented_image_path = f"{output_image_dir}/{image_name}_aug_{i+1}{ext}"
        augmented_mask_path = f"{output_mask_dir}/{image_name}_aug_{i+1}{ext}"
        
        cv2.imwrite(augmented_image_path, augmented_image)
        cv2.imwrite(augmented_mask_path, augmented_mask)

def augment_images_and_masks_in_directory(image_dir, mask_dir, output_image_dir, output_mask_dir, num_augmented_images=10):
    """
    Apply augmentation to all images and masks in the given directories and generate specified number of augmented versions for each.
    """
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename)  # Assuming mask has the same filename
            augment_image_and_mask(image_path, mask_path, output_image_dir, output_mask_dir, num_augmented_images=num_augmented_images)

# Example usage with your specific paths
image_dir = r'/mnt/c/Users/matys/Desktop/FIT/3BIT/zimni/BC-work/ContaminationMeasurement/WholeDataset'
output_image_dir = r'/mnt/c/Users/matys/Desktop/FIT/3BIT/zimni/BC-work/ContaminationMeasurement/augmented/images'
mask_dir = r'/mnt/c/Users/matys/Desktop/FIT/3BIT/zimni/BC-work/ContaminationMeasurement/masks_grayscale'
output_mask_dir = r'/mnt/c/Users/matys/Desktop/FIT/3BIT/zimni/BC-work/ContaminationMeasurement/augmented/masks'
augment_images_and_masks_in_directory(image_dir, mask_dir, output_image_dir, output_mask_dir, 10)
