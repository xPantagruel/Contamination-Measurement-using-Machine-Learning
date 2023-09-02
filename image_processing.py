import os
import glob
import cv2
import numpy as np

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

def horizonta_edge_filter(image):
    # Define the horizontal filter kernel
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Apply the filter using cv2.filter2D()
    filtered_image = cv2.filter2D(image, -1, kernel)

    # Display the original and filtered images
    cv2.imshow('Original Image', image)
    cv2.imshow('Filtered Image', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def find_gradients(image):
    # Compute the gradients using the Sobel operator
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=31)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=31)

    # Compute the gradient magnitude and direction
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Normalize the gradient magnitude for visualization
    gradient_magnitude_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


    # Display the original image and normalized gradient magnitude
    # Get the screen dimensions
    screen_width, screen_height = (1920, 1080)  # Replace with your screen resolution

    # Create a window with the screen dimensions
    cv2.namedWindow('gradient_magnitude_normalized', cv2.WINDOW_NORMAL)
    
    # Resize the window to fit the screen
    cv2.resizeWindow('gradient_magnitude_normalized', screen_width, screen_height)
    
    # Display the image
    cv2.imshow('gradient_magnitude_normalized', gradient_magnitude_normalized)
    
def post_process_contamination(detected_contamination):
    pass
