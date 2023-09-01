import os
import glob

# creates list of image paths from folder
def load_images_from_folder(folder_path):
    # Define a list to store image file paths
    image_paths = []
    # Use the 'glob' function to search for image files in the directory and its subdirectories
    # You can specify different image file extensions like '*.jpg', '*.png', '*.jpeg', etc.
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', "*.tif"]  # Add more extensions as needed

    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, '**', extension), recursive=True))
    
    return image_paths

def load_image(image_path):
    pass

def preprocess_image(image):
    pass

def post_process_contamination(detected_contamination):
    pass
