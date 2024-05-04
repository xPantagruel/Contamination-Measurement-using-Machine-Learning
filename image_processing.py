import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from statistics import mode
from scipy.signal import find_peaks
DEBUG = False

def find_contamination_bottom_and_top(image, starting_point, position=0, num_rows=50, shouwDebug=False):
    # Get the dimensions of the image
    height, width = image.shape

    # Extract the vertical line at the given position
    if position == 0:
        line_x_position = width // 2
    else:
        line_x_position = position

    # Define the range of rows to consider around the initial position
    start_row = max(0, line_x_position - num_rows)
    end_row = min(width - 1, line_x_position + num_rows) + 30 # add more columns in direction to right from middle of contamination

    # Extract pixel values from the selected rows
    line_values = np.mean(image[:, start_row:end_row + 1], axis=1)

    # use gausian window to smooth the line values
    line_values = np.convolve(line_values, np.ones(5) / 5, mode='same')

    # Calculate the first derivative of the line values
    line_first_gradient = np.gradient(line_values)
    line_first_derivative = np.diff(line_values)

    # Find all local maximums and minimums in the first derivative
    maxs, _ = find_peaks(line_first_gradient, prominence=0)  # You can adjust prominence as needed
    mins, _ = find_peaks(-line_first_gradient, prominence=0)  # Invert and find peaks for minima

    # remove maxs and mins that are starting_point + 100 > and <
    # Filter out maxs and mins that are outside the specified range
    window = 180
    start_index = max(0, starting_point - window)
    end_index = min(height - 1, starting_point + window)
    maxs = [max_point for max_point in maxs if start_index <= max_point <= end_index]
    mins = [min_point for min_point in mins if start_index <= min_point <= end_index]
                
    # Find the biggest maximum in the direction where y > starting_point
    bottom_of_contamination = None
    max_value = float('-inf')
    for max_point in maxs:
        if max_point > starting_point and line_first_gradient[max_point] > max_value:
            bottom_of_contamination = max_point
            max_value = line_first_gradient[max_point]
    
    # Find the top of contamination, where y < starting_point and exhibits the biggest change from minimum to maximum towards the origin on the y-axis
    top_of_contamination = None
    max_difference = 0

    # Iterate from the starting point to the starting point - 200
    for i in range(bottom_of_contamination, bottom_of_contamination - 200, -1):
        # find the closest maximum from the minimum in direction of y < starting_point
        if i in mins:
            # go through the maximums and find the closest one to the minimum in direction of j < i and calculate the difference between them and if the difference is bigger than the previous one, set the top of contamination to the minimum and go to next minimum and do the same
            for MaxN in maxs:
                if MaxN in maxs and MaxN < i : 
                    difference = line_first_gradient[MaxN] - line_first_gradient[i]
                    if difference > max_difference:
                        top_of_contamination = i
                        max_difference = difference
                    break
    
    if shouwDebug:
        # Plot the first derivative of the line profile with maximums and minimums
        plt.figure(figsize=(15, 5))

        # Plot the original image with highlighted points
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        if top_of_contamination is not None:
            plt.axhline(y=top_of_contamination, color='m', linestyle='--', linewidth=1.5)
        if bottom_of_contamination is not None:
            plt.axhline(y=bottom_of_contamination, color='b', linestyle='--', linewidth=1.5)
        if starting_point is not None:
            plt.axhline(y=starting_point, color='y', linestyle='--', linewidth=1.5)
        # show start and end of selected columns
        plt.axvline(x=start_row, color='r', linestyle=':', linewidth=1.5)
        plt.axvline(x=end_row, color='r', linestyle=':', linewidth=1.5)
        plt.title('ROI', fontsize=22)

        # Plot the first derivative of the line profile
        plt.subplot(1, 2, 2)
        plt.plot(range(height), line_first_gradient)
        # plt.scatter(maxs, [line_first_gradient[i] for i in maxs], color='r', label='Max')
        # plt.scatter(mins, [line_first_gradient[i] for i in mins], color='g', label='Min')
        if bottom_of_contamination is not None:
            plt.scatter(bottom_of_contamination, line_first_gradient[bottom_of_contamination], color='b', label='Bottom of Contamination')
        if top_of_contamination is not None:
            plt.scatter(top_of_contamination, line_first_gradient[top_of_contamination], color='m', label='Top of Contamination')

        if starting_point is not None:
            plt.scatter(starting_point, line_first_gradient[starting_point], color='y', label='Starting Point')
        plt.xlabel('Y Axis (Height of Image)', fontsize=18)
        plt.ylabel('Derivative Value', fontsize=18)  # More precise labeling
        plt.title('Gradient Analysis of Vertical Line Profile', fontsize=22)  # Updated for clarity
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Return the found maximums and minimums
    return maxs, mins, bottom_of_contamination, top_of_contamination


def blurring(img):
    return apply_gaussian_blur(img)


def apply_gaussian_blur(image, kernel_size=(5, 5), sigma_x=0):
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma_x)
    return blurred_image


def apply_median_blur(image, kernel_size=5):
    blurred_image = cv2.medianBlur(image, kernel_size)
    return blurred_image

def get_starting_point_TEST(image, showDebug=False, TinBallEdge=0):
    # Compute the vertical profile by averaging pixel values along the horizontal axis in middle
    column_start = image.shape[1] // 2

    if column_start + 100 > image.shape[1] or column_start - 100 < 0:
        return -1

    vertical_profile = np.mean(image[:, column_start - 100:column_start + 100], axis=1)
    
    starting_point = -1
    
    # Find all local maximas and minimas 
    max_peaks, _ = find_peaks(vertical_profile)
    min_peaks, _ = find_peaks(-vertical_profile)
    
    # add to max peaks the last point of the vertical profile
    max_peaks = np.append(max_peaks, len(vertical_profile) - 1)
    # Find potential starting points
    starting_points = []
    threshold_value = 40  # Threshold value for the difference between max and min
    # range 40 to 0 
    for threshold_value in range(threshold_value, 0, -1):
        for min_point in min_peaks:
            for max_point in max_peaks:
                if min_point < max_point and vertical_profile[max_point] - vertical_profile[min_point] > threshold_value:
                    starting_points.append(min_point)
                    break
        if starting_points != []:
            break
    
    # Choose the starting point as the highest minimum
    # remove starting points that are not in the range bigger than tin ball edge
    starting_points = [point for point in starting_points if point > TinBallEdge]
    
    starting_point = max(starting_points) if starting_points else -1
    
    # show the graph with all maximums and minimums
    if showDebug :
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))

        # Plot original image
        axs[0].imshow(image, cmap='gray')
        axs[0].axvline(x=column_start, color='r', linestyle='--', label='Column Start')
        axs[0].axhline(y=starting_point, color='g', linestyle='--', label='Starting Point')
        axs[0].set_xlabel('X Axis', fontsize=18)
        axs[0].set_ylabel('Y Axis', fontsize=18)
        axs[0].set_title('Original Image', fontsize=18)
        axs[0].legend()

        # Plot vertical profile
        axs[1].plot(vertical_profile, color='b')
        axs[1].scatter(starting_point, vertical_profile[starting_point], color='g', label='Starting Point')
        axs[1].set_xlabel('Y Axis (Height of Image)', fontsize=18)
        axs[1].set_ylabel('Pixel Value', fontsize=18)
        axs[1].set_title('Vertical Profile', fontsize=18)
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    return starting_point

def get_Roi(image, left_boundary, right_boundary):
    # Get image dimensions
    height, width= image.shape

    # Crop the image based on the provided boundaries
    cropped_img = image[0:height, left_boundary:right_boundary]

    return cropped_img

def apply_closing(img, kernel_size=(5, 5)):
    # Convert the image to grayscale if it's not already in grayscale
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define the kernel for morphological operations (structuring element)
    kernel = np.ones(kernel_size, np.uint8)

    # Perform the closing operation
    closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return closed_img

def apply_opening(img, kernel_size=(5, 5)):
    # Convert the image to grayscale if it's not already in grayscale
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define the kernel for morphological operations (structuring element)
    kernel = np.ones(kernel_size, np.uint8)

    # Perform the opening operation
    opened_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    return opened_img

def get_contamination_range(image, maxY):
    img = image.copy()

    # Get the height and width of the image
    height, width = img.shape

    # Initialize variables to store the y values for each direction
    center_x = width // 2  # Start from the center column
    left_x = center_x - 400  # 400 columns to the left
    right_x = center_x + 400  # 400 columns to the right

    def findEdgeToRight(middle, x_end):
        end_x = -1  # Initialize the end position
        x = middle
        while x < x_end:
            x += 1
            y = height - 1  # Start from the bottom row
            while y >= maxY:
                y -= 1
                if y < maxY:
                    return end_x

                pixel_value = img[y, x]
                if pixel_value > 150 and y > maxY:
                    end_x = x
                    break

        return end_x

    def findEdgeToLeft(x_start, middle):
        end_x = -1  # Initialize the end position
        x = middle
        while x > x_start:
            x -= 1
            y = height - 1  # Start from the bottom row
            while y >= maxY:
                y -= 1
                if y < maxY:
                    return end_x

                pixel_value = img[y, x]
                if pixel_value > 150 and y > maxY:
                    end_x = x
                    break

        return end_x

    # Find the start and end positions of contamination while moving to the right
    right_end = findEdgeToRight(center_x, right_x + 1)

    # Find the start and end positions of contamination while moving to the left
    left_end = findEdgeToLeft(left_x, center_x + 1)
    middle = ((right_end - left_end) // 2) + left_end

    if DEBUG:
        print("Start of contamination:", left_end)
        print("End of contamination:", right_end)

        print("width:", right_end - left_end)
        print("middle:", middle)

    return left_end, right_end, middle

def get_mode_height_of_tin_ball_left_side(image):
    img = image.copy()

    # Get the height and width of the image
    height, width = img.shape

    # Initialize variables to keep track of the minimum pixel value and its coordinates
    min_pixel_values = []
    pixel_height = 0

    # Iterate through the first 10 columns and all rows from bottom to top
    for x in range(10):  # First 10 columns
        for y in range(height - 1, -1, -1):  # Iterate from the bottom to the top
            pixel_value = img[y, x]
            if pixel_value > 150:
                min_pixel_values.insert(0, y)
                break

    if (min_pixel_values):
        # median from all
        pixel_height = mode(min_pixel_values)
    return pixel_height

def median_blur(image, kernel_size):
    # Apply mdian blur filter
    blurred_image = cv2.medianBlur(image, kernel_size)

    return blurred_image

def get_mode_height_of_tin_ball_right_side(image):
    img = image.copy()

    # Get the height and width of the image
    height, width = img.shape

    # Initialize a list to store the pixel heights of white pixels
    white_pixel_heights = []

    # Iterate through the last 10 columns and all rows from the bottom to the top
    for x in range(width - 10, width):  # Last 10 columns
        for y in range(height - 1, -1, -1):  # Iterate from the bottom to the top
            pixel_value = img[y, x]
            if pixel_value > 150: 
                white_pixel_heights.append(y)
                break

    if white_pixel_heights:
        # Calculate the mode (the value that appears most frequently) of the white pixel heights
        mode_height = max(set(white_pixel_heights),
                          key=white_pixel_heights.count)
    else:
        mode_height = -1  

    return mode_height

def get_mode_height_of_tin_ball_left_side(image):
    img = image.copy()

    # Get the height and width of the image
    height, width = img.shape

    # Initialize variables to keep track of the minimum pixel value and its coordinates
    min_pixel_values = []
    pixel_height = 0

    # Iterate through the first 10 columns and all rows from bottom to top
    for x in range(10):  # First 10 columns
        for y in range(height - 1, -1, -1):  # Iterate from the bottom to the top
            pixel_value = img[y, x]
            if pixel_value > 150:
                min_pixel_values.insert(0, y)
                break

    if (min_pixel_values):
        # median from all
        pixel_height = mode(min_pixel_values)
    return pixel_height

def median_blur(image, kernel_size):
    # Apply mdian blur filter
    blurred_image = cv2.medianBlur(image, kernel_size)

    return blurred_image

def get_mode_height_of_tin_ball_right_side(image):
    img = image.copy()

    # Get the height and width of the image
    height, width = img.shape

    # Initialize a list to store the pixel heights of white pixels
    white_pixel_heights = []

    # Iterate through the last 10 columns and all rows from the bottom to the top
    for x in range(width - 10, width):  # Last 10 columns
        for y in range(height - 1, -1, -1):  # Iterate from the bottom to the top
            pixel_value = img[y, x]
            if pixel_value > 150: 
                white_pixel_heights.append(y)
                break

    if white_pixel_heights:
        # Calculate the mode (the value that appears most frequently) of the white pixel heights
        mode_height = max(set(white_pixel_heights),
                          key=white_pixel_heights.count)
    else:
        mode_height = -1  

    return mode_height

def plot_histogram(image_path):
    # Load the image in grayscale
    image = load_image(image_path)
    # preprocessed_image = preprocess_image(image)
    preprocessed_image = median_blur(image, 5)
    # Calculate the histogram
    hist = cv2.calcHist([preprocessed_image], [0], None, [256], [0, 256])

    # Create a figure with two subplots: image and histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the image
    ax1.imshow(preprocessed_image, cmap='gray')
    ax1.set_title('Image')
    ax1.axis('off')

    # Plot the histogram
    ax2.plot(hist, color='black')
    ax2.set_title('Histogram')
    ax2.set_xlabel('Pixel Value')
    ax2.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()
    return 0,0

def otsu_thresholding(image):
    # Apply Otsu's thresholding
    _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresholded

def load_image(image_path):
    # resize image to 1024x768
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # check if dimensions are 1024x768
    # if image.shape[0] != 768 or image.shape[1] != 1024:
    #     image = cv2.resize(image, (1024, 768))

    return image

def load_images_from_folder(folder_path):
    image_paths = []
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif',
                        '*.bmp', "*.tif"] # Supported image extensions

    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(
            folder_path, '**', extension), recursive=True))

    return image_paths

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
    gradient_magnitude_image = gradient_magnitude(
        gradient_x_image, gradient_y_image)
    suppressed_image = non_maximum_suppression(
        gradient_magnitude_image, gradient_x_image, gradient_y_image)
    strong_edges, weak_edges = hysteresis_threshold(
        suppressed_image, low_threshold, high_threshold)
    edge_image = edge_tracking_by_hysteresis(strong_edges, weak_edges)
    return edge_image