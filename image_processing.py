import os
import glob
import cv2
from skimage import color, filters
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_laplace
from PIL import Image

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
 
def laplacian_of_gaussian(gray_image, sigma=1):
    # Apply the Laplacian of Gaussian (LoG) filter
    log_edges = gaussian_laplace(gray_image, sigma=sigma)
    
    # Adjust the range of values for visualization
    log_edges = (log_edges - np.min(log_edges)) / (np.max(log_edges) - np.min(log_edges)) * 255
    
    # Convert to 8-bit unsigned integer (0-255)
    log_edges = np.uint8(log_edges)
    
    return log_edges

def hough_transform_line_detection(image):
    pass

def gaussian_smoothing(image, kernel_size=5, sigma=1.4):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def gradient_magnitude(dx, dy):
    return np.sqrt(dx ** 2 + dy ** 2)

def gradient_x(image):
    return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)

def gradient_y(image):
    return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

def non_maximum_suppression(magnitude, gradient_x, gradient_y):
    height, width = magnitude.shape
    suppressed = np.zeros((height, width), dtype=np.uint8)
    angle = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
    angle[angle < 0] += 180

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            q1, q2 = 255, 255
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q1 = magnitude[i, j+1]
                q2 = magnitude[i, j-1]
            elif 22.5 <= angle[i, j] < 67.5:
                q1 = magnitude[i+1, j-1]
                q2 = magnitude[i-1, j+1]
            elif 67.5 <= angle[i, j] < 112.5:
                q1 = magnitude[i+1, j]
                q2 = magnitude[i-1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q1 = magnitude[i-1, j-1]
                q2 = magnitude[i+1, j+1]

            if magnitude[i, j] >= q1 and magnitude[i, j] >= q2:
                suppressed[i, j] = magnitude[i, j]

    return suppressed

def hysteresis_threshold(image, low_threshold, high_threshold):
    strong_edges = (image >= high_threshold)
    weak_edges = (image >= low_threshold) & (image < high_threshold)
    return strong_edges, weak_edges

def edge_tracking_by_hysteresis(strong_edges, weak_edges):
    height, width = strong_edges.shape
    edge_image = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if strong_edges[i, j]:
                edge_image[i, j] = 255
            elif weak_edges[i, j]:
                neighbors = edge_image[i-1:i+2, j-1:j+2]
                if np.any(neighbors == 255):
                    edge_image[i, j] = 255

    return edge_image

def canny_edge_detection(image, low_threshold, high_threshold, kernel_size=5, sigma=1.4):
    smoothed_image = gaussian_smoothing(image, kernel_size, sigma)
    gradient_x_image = gradient_x(smoothed_image)
    gradient_y_image = gradient_y(smoothed_image)
    gradient_magnitude_image = gradient_magnitude(gradient_x_image, gradient_y_image)
    suppressed_image = non_maximum_suppression(gradient_magnitude_image, gradient_x_image, gradient_y_image)
    strong_edges, weak_edges = hysteresis_threshold(suppressed_image, low_threshold, high_threshold)
    edge_image = edge_tracking_by_hysteresis(strong_edges, weak_edges)
    # Display the Canny edges image
    # cv2.namedWindow('Canny Edges', cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty('Canny Edges', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.imshow('Canny Edges', edge_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return edge_image

def plot_histogram(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate the histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(image.ravel(), 256, [0, 256])
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.show()

def blurring(img):
    return apply_gaussian_blur(img)

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma_x=0):
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma_x)
    # cv2.namedWindow('Blurred Image', cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty('Blurred Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.imshow('Blurred Image', blurred_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return blurred_image

def apply_median_blur(image, kernel_size = 5):
    blurred_image = cv2.medianBlur(image, kernel_size)
    return blurred_image

def apply_bilateral_filter(image, d=9,simgaColor=75,sigmaSpace=75):#this One looks like working the best
    blurred_image = cv2.bilateralFilter(image, d, simgaColor, sigmaSpace)
    return blurred_image

def apply_fastNlMeansDenoisingColored(image ):
    pass

def thresholding(img):
    ret, thresh0 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)#VERY GOOD 
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)#For bottom of layer might be top
    ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)

    # Displaying the output image
    # cv2.namedWindow('Threshold', cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty('Threshold', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.imshow('Threshold', thresh2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return thresh2

def sharpening_image(image):
    # Define the sharpening kernel
    sharpening_kernel = np.array([[-1, -1, -1],
                                [-1, 9, -1],
                                [-1, -1, -1]])
    
    sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
    return sharpened_image

def post_process_contamination(detected_contamination):
    pass
