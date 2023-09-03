import cv2
from skimage import color
import matplotlib.pyplot as plt
from image_processing import *

def visualize_edge_detection(image):
    # Apply Sobel operator
    sobel_edges = sobel_edge_detection(image)
    
    # Apply Prewitt operator
    prewitt_edges = prewitt_edge_detection(image)
    
    # Apply Roberts operator
    roberts_edges = roberts_edge_detection(image)
    
    # Apply Canny edge detector
    canny_edges = canny_edge_detection(image)
    
    # Apply Laplacian of Gaussian
    log_edges = laplacian_of_gaussian(image)
    
    # Create a subplot for each edge detection method
    plt.figure(figsize=(15, 10))
    
    # Display the source image
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Source Image')
    
    # Display Sobel edges
    plt.subplot(2, 3, 2)
    plt.imshow(sobel_edges, cmap='gray')
    plt.title('Sobel Operator')
    
    # Display Prewitt edges
    plt.subplot(2, 3, 3)
    plt.imshow(prewitt_edges, cmap='gray')
    plt.title('Prewitt Operator')
    
    # Display Roberts edges
    plt.subplot(2, 3, 4)
    plt.imshow(roberts_edges, cmap='gray')
    plt.title('Roberts Operator')
    
    # Display Canny edges
    plt.subplot(2, 3, 5)
    plt.imshow(canny_edges, cmap='gray')
    plt.title('Canny Edge Detector')
    
    # Display Laplacian of Gaussian edges
    plt.subplot(2, 3, 6)
    plt.imshow(log_edges, cmap='gray')
    plt.title('Laplacian of Gaussian')
    
    # Show the plots
    plt.tight_layout()
    plt.show()