import os
import glob

# creates list of image paths from folder
def load_images_from_folder(folder_path):
    image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))
    return image_paths

def load_image(image_path):
    pass

def preprocess_image(image):
    pass

def post_process_contamination(detected_contamination):
    pass
