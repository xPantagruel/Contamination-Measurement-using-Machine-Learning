import os
import glob
import cv2
from skimage import color, filters
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_laplace

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
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# cutt off the edges of the image
def preprocess_image(image):
    return image[100:image.shape[0] - 100, 100:image.shape[1] - 200]
    # return image.crop((100, 100, image.width - 100, image.height - 200))

def sobel_edge_detection(gray_image):

    # Apply Sobel operator
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine the gradients to find edges
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    
    return magnitude

def prewitt_edge_detection(gray_image):
    # Apply Prewitt operator
    prewitt_edges = filters.prewitt(gray_image)
    
    return prewitt_edges

def roberts_edge_detection(gray_image):
    # Apply Roberts operator
    roberts_edges = filters.roberts(gray_image)
    
    return roberts_edges
 
def canny_edge_detection(gray_image):
    # Apply Canny edge detector
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
    
    return edges

def laplacian_of_gaussian(gray_image, sigma=1):
    # Apply the Laplacian of Gaussian (LoG) filter
    log_edges = gaussian_laplace(gray_image, sigma=sigma)
    
    # Adjust the range of values for visualization
    log_edges = (log_edges - np.min(log_edges)) / (np.max(log_edges) - np.min(log_edges)) * 255
    
    # Convert to 8-bit unsigned integer (0-255)
    log_edges = np.uint8(log_edges)
    
    return log_edges

def post_process_contamination(detected_contamination):
    pass
