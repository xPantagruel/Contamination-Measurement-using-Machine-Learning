import cv2
from skimage import color
import matplotlib.pyplot as plt
from image_processing import *
from PIL import Image

def show_images(images, names):
    """
    Display a list of images with corresponding names using Matplotlib.

    Args:
        images (list): List of image data (numpy arrays or PIL images).
        names (list): List of image names (strings).
    """
    if len(images) != len(names):
        raise ValueError("The number of images must be equal to the number of names.")

    num_images = len(images)
    rows = (num_images // 3) + 1  # Display 3 images per row

    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    axes = axes.ravel()

    for i in range(num_images):
        axes[i].imshow(images[i])
        axes[i].set_title(names[i])
        axes[i].axis('off')

    # Hide any remaining empty subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def create_imagelist_edge_detection(image):
    # Example usage:
    images_list = [
        cv2.convertScaleAbs(image),
        cv2.convertScaleAbs(sobel_edge_detection(image)),
        cv2.convertScaleAbs(prewitt_edge_detection(image)),
        cv2.convertScaleAbs(roberts_edge_detection(image)),
        cv2.convertScaleAbs(canny_edge_detection(image)),
        cv2.convertScaleAbs(laplacian_of_gaussian(image)),
        cv2.convertScaleAbs(hough_transform_line_detection(image))
    ]

    titles_list = [
        'source',
        'Sobel Operator',
        'Prewitt Operator',
        'Roberts Operator',
        'Canny Edge Detector',
        'Laplacian of Gaussian',
        'hough transform'
    ]

    show_images(images_list, titles_list)