import os
import glob
import cv2
from skimage import color, filters
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_laplace
from PIL import Image
from statistics import mode
from scipy.signal import find_peaks

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
    end_row = min(width - 1, line_x_position + num_rows)

    # Extract pixel values from the selected rows
    line_values = np.mean(image[:, start_row:end_row + 1], axis=1)

    # Calculate the first derivative of the line values
    line_first_derivative = np.gradient(line_values)
    
    # Find all local maximums and minimums in the first derivative
    # TODO I calculate the maxs and mins for whole graph but I can instead only the window arround the starting point 
    maxs, _ = find_peaks(line_first_derivative, prominence=0)  # You can adjust prominence as needed
    mins, _ = find_peaks(-line_first_derivative, prominence=0)  # Invert and find peaks for minima

    # remove maxs and mins that are starting_point + 100 > and <
    # Filter out maxs and mins that are outside the specified range
    window = 160
    start_index = max(0, starting_point - window)
    end_index = min(height - 1, starting_point + window)
    maxs = [max_point for max_point in maxs if start_index <= max_point <= end_index]
    mins = [min_point for min_point in mins if start_index <= min_point <= end_index]
                
    # Find the biggest maximum in the direction where y > starting_point
    bottom_of_contamination = None
    max_value = float('-inf')
    for max_point in maxs:
        if max_point > starting_point and line_first_derivative[max_point] > max_value:
            bottom_of_contamination = max_point
            max_value = line_first_derivative[max_point]
    
    # Find the top of contamination, where y < starting_point and exhibits the biggest change from minimum to maximum towards the origin on the y-axis
    top_of_contamination = None
    max_difference = 0

    # Iterate from the starting point to the starting point - 200
    for i in range(starting_point, starting_point - 200, -1):
        # find the closest maximum from the minimum in direction of y < starting_point
        if i in mins:
            # go through the maximums and find the closest one to the minimum in direction of j < i and calculate the difference between them and if the difference is bigger than the previous one, set the top of contamination to the minimum and go to next minimum and do the same
            for MaxN in maxs:
                if MaxN in maxs and MaxN < i : 
                    difference = line_first_derivative[MaxN] - line_first_derivative[i]
                    if difference > max_difference:
                        top_of_contamination = i
                        max_difference = difference
                    break
    
    if shouwDebug:
        # Plot the first derivative of the line profile with maximums and minimums
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')

        if top_of_contamination is not None:
            plt.axhline(y=top_of_contamination, color='g', linestyle='--', linewidth=1.5)
        if bottom_of_contamination is not None:
            plt.axhline(y=bottom_of_contamination, color='r', linestyle='--', linewidth=1.5)
        # show start and end of selected columns
        plt.axvline(x=start_row, color='r', linestyle='--', linewidth=1.5)
        plt.axvline(x=end_row, color='r', linestyle='--', linewidth=1.5)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(height), line_first_derivative)
        plt.scatter(maxs, [line_first_derivative[i] for i in maxs], color='r', label='Max')
        plt.scatter(mins, [line_first_derivative[i] for i in mins], color='g', label='Min')
        if bottom_of_contamination is not None:
            plt.scatter(bottom_of_contamination, line_first_derivative[bottom_of_contamination], color='b', label='Bottom of Contamination')
        if top_of_contamination is not None:
            plt.scatter(top_of_contamination, line_first_derivative[top_of_contamination], color='m', label='Top of Contamination')        

        
        plt.xlabel('Y Axis (Height of Image)')
        plt.ylabel('First Derivative')
        plt.title('First Derivative of Vertical Line Profile')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Return the found maximums and minimums
    return maxs, mins, bottom_of_contamination, top_of_contamination

def find_contamination_height(image, starting_point, position=200, num_rows=10, shouwDebug=False):
    # Get the dimensions of the image
    height, width = image.shape

    # Extract the vertical line at the given position
    if position == 0:
        line_x_position = width // 2
    else:
        line_x_position = position

    # Define the range of rows to consider around the initial position
    start_row = max(0, line_x_position - num_rows)
    end_row = min(width - 1, line_x_position + num_rows)

    # Extract pixel values from the selected rows
    line_values = np.mean(image[:, start_row:end_row + 1], axis=1)

    # Calculate the first derivative of the line values
    line_first_derivative = np.gradient(line_values)

    # find maximums and minimums around starting point in vertical profile first derivation of image about 300 around starting point
    maxs = []
    mins = []
    for i in range(starting_point - 100, starting_point + 100):
        if i > 0 and i < height - 1:
            if line_first_derivative[i] > line_first_derivative[i - 1] and line_first_derivative[i] > line_first_derivative[i + 1]:
                maxs.append(i)
            if line_first_derivative[i] < line_first_derivative[i - 1] and line_first_derivative[i] < line_first_derivative[i + 1]:
                mins.append(i)
                
    # Find the biggest maximum in the direction where y > starting_point
    bottom_of_contamination = None
    max_value = float('-inf')
    for max_point in maxs:
        if max_point > starting_point and line_first_derivative[max_point] > max_value:
            bottom_of_contamination = max_point
            max_value = line_first_derivative[max_point]
    
    # Plot the first derivative of the line profile with maximums and minimums
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    if bottom_of_contamination is not None:
        plt.axhline(y=bottom_of_contamination, color='r', linestyle='--', linewidth=1.5)
    plt.subplot(1, 2, 2)
    plt.plot(range(height), line_first_derivative)
    plt.scatter(maxs, [line_first_derivative[i] for i in maxs], color='r', label='Max')
    plt.scatter(mins, [line_first_derivative[i] for i in mins], color='g', label='Min')
    if bottom_of_contamination is not None:
        plt.scatter(bottom_of_contamination, line_first_derivative[bottom_of_contamination], color='b', label='Biggest Max')        
        
    plt.xlabel('Y Axis (Height of Image)')
    plt.ylabel('First Derivative')
    plt.title('First Derivative of Vertical Line Profile')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Return the found maximums and minimums
    return maxs, mins, bottom_of_contamination

def get_starting_point_TEST(image, column_start=200, showDebug=False, TinBallEdge=0):
    # Compute the vertical profile by averaging pixel values along the horizontal axis in middle only 50 pixels in each direction 
    vertical_profile = np.mean(image[:, column_start - 100:column_start + 100], axis=1)
    # vertical_profile = np.mean(image, axis=1)
    starting_point = -1
    # find all maximums and minimums 
    maxs = []
    mins = []
    for i in range(1, len(vertical_profile) - 1):
        if vertical_profile[i] > vertical_profile[i - 1] and vertical_profile[i] > vertical_profile[i + 1]:
            maxs.append(i)
        if vertical_profile[i] < vertical_profile[i - 1] and vertical_profile[i] < vertical_profile[i + 1]:
            mins.append(i)

    # Find potential starting points
    starting_points = []
    for min_point in mins:
        for max_point in maxs:
            if min_point < max_point and vertical_profile[max_point] - vertical_profile[min_point] > 40:
                starting_points.append(min_point)
                break

    # Choose the starting point as the highest minimum
    # TODO TEST IT 
    # remove starting points that are not in the range bigger than tin ball edge
    starting_points = [point for point in starting_points if point > TinBallEdge]
    
    starting_point = max(starting_points) if starting_points else -1
    
    # show the graph with all maximums and minimums
    if showDebug :
        plt.plot(vertical_profile, color='b')
        # show starting point
        plt.scatter(starting_point, vertical_profile[starting_point], color='r', label='Starting Point')
        # plt.scatter(maxs, [vertical_profile[i] for i in maxs], color='r', label='Max')
        # plt.scatter(mins, [vertical_profile[i] for i in mins], color='g', label='Min')
        plt.xlabel('Vertical Position')
        plt.ylabel('Pixel Value')
        plt.title('Vertical Profile with Maxs and Mins')
        plt.legend()
        plt.show()

    return starting_point

def get_starting_point(image, column_start=200, showDebug=False,TinBallEdge=0):
    # Compute the vertical profile by averaging pixel values along the horizontal axis
    vertical_profile = np.mean(image, axis=1)

    # Find all minima in the second half of the vertical profile
    max_height = len(vertical_profile)
    minima_indices = []
    for i in range(max_height // 2, max_height):
        if i > max_height // 2 and i < max_height - 1:
            if vertical_profile[i] < vertical_profile[i - 1] and vertical_profile[i] < vertical_profile[i + 1]:
                minima_indices.append(i)

    # Initialize variables for tracking the best pair of minima
    max_difference = 0
    best_pair = None

    # Iterate through the minima to find the pair with the biggest difference within 250 pixels
    for i in range(len(minima_indices)):
        for j in range(i+1, len(minima_indices)):
            if abs(minima_indices[i] - minima_indices[j]) <= 250:
                difference = vertical_profile[minima_indices[i]] - vertical_profile[minima_indices[j]]
                if difference < max_difference:
                    max_difference = difference
                    best_pair = (minima_indices[i], minima_indices[j])

    # Choose the starting point as the minimum of the pair with the biggest difference
    if best_pair is not None:
        starting_point = min(best_pair)
    else:
        starting_point = -1

        
    if showDebug :
        # Plot the vertical profile with the starting point and the best pair of minima
        plt.plot(vertical_profile, color='b')
        plt.scatter(starting_point, vertical_profile[starting_point], color='r', label='Starting Point')
        if best_pair is not None:
            plt.scatter(best_pair[0], vertical_profile[best_pair[0]], color='g', label='Minima 1')
            plt.scatter(best_pair[1], vertical_profile[best_pair[1]], color='m', label='Minima 2')
        plt.xlabel('Vertical Position')
        plt.ylabel('Pixel Value')
        plt.title('Vertical Profile with Starting Point and Best Minima Pair')
        plt.legend()
        plt.show()

    # # Plot the vertical profile with all minima marked
    # plt.plot(vertical_profile, color='b')
    # plt.scatter(minima_indices, [vertical_profile[i] for i in minima_indices], color='r', label='Minima')
    # plt.xlabel('Vertical Position')
    # plt.ylabel('Pixel Value')
    # plt.title('Vertical Profile with Minima in the Second Half')
    # plt.legend()
    # plt.show()

    return starting_point


def plot_mean_pixel_values(image, num_columns=200):
    # Get the middle of the image
    middle_x = image.shape[1] // 2

    # Extract 10 columns starting from the middle
    start_column = middle_x - (num_columns // 2)
    end_column = start_column + num_columns
    extracted_columns = image[:, start_column:end_column]

    # Calculate the mean pixel values across columns
    mean_pixel_values = np.mean(extracted_columns, axis=1)

    # Plot the image
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')

    # Plot the mean pixel values
    ax2.plot(mean_pixel_values)
    ax2.set_title('Mean Pixel Values Across Rows')
    ax2.set_xlabel('Row')
    ax2.set_ylabel('Mean Pixel Value')

    plt.tight_layout()
    plt.show()

def get_Roi(image, left_boundary, right_boundary):
    # Get image dimensions
    height, width= image.shape

    # Crop the image based on the provided boundaries
    cropped_img = image[0:height, left_boundary:right_boundary]

    return cropped_img

def plot_different_x_positions_with_graph(gray_img):
    # Get the dimensions of the image
    height, width = gray_img.shape

    # Define x positions for the lines (10 different x positions around the middle)
    x_positions = [width // 2 - 50, width // 2 - 40, width // 2 - 30, width // 2 - 20,
                   width // 2 - 10, width // 2, width // 2 + 10, width // 2 + 20,
                   width // 2 + 30, width // 2 + 40]

    # Extract the pixel values along different x positions
    line_values = {}
    for x in x_positions:
        line_values[x] = [gray_img[y, x] for y in range(height)]

    # Plotting the graph with vertical lines and the combined graph
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the grayscale image with vertical lines
    ax1.imshow(gray_img, cmap='gray')
    for x in x_positions:
        ax1.axvline(x=x, color='red', linestyle='--', linewidth=1.5)
    ax1.axis('off')
    ax1.set_title('Grayscale Image with Vertical Lines')

    # Plot the combined graph for all vertical lines
    for x in x_positions:
        ax2.plot(line_values[x], label=f'x={x}')
    ax2.set_xlabel('Y Axis (Height of Image)')
    ax2.set_ylabel('Pixel Value')
    ax2.set_title('Combined Vertical Line Profiles')
    ax2.legend()

    plt.tight_layout()
    plt.show()
    
def plot_vertical_line_cv2(gray_img, position=0, num_rows=10, shouwDebug=False):
    # Get the dimensions of the image
    height, width = gray_img.shape

    # Extract the vertical line at the given position
    if position == 0:
        line_x_position = width // 2
    else:
        line_x_position = position

    # Define the range of rows to consider around the initial position
    start_row = max(0, line_x_position - num_rows)
    end_row = min(width - 1, line_x_position + num_rows)

    # Extract pixel values from the selected rows
    line_values = np.mean(gray_img[:, start_row:end_row + 1], axis=1)

    # Calculate the first derivative of the line values
    line_first_derivative = np.gradient(line_values)

    if shouwDebug:
        # Plotting the graph and image with the line
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Plot the grayscale image
        ax1.imshow(gray_img, cmap='gray')
        ax1.plot([line_x_position, line_x_position], [0, height - 1], color='red', linestyle='--', linewidth=1.5)
        ax1.axis('off')
        ax1.set_title('Grayscale Image with Vertical Line')

        # Plot the line profile
        ax2.plot(range(height), line_values)
        ax2.set_xlabel('Y Axis (Height of Image)')
        ax2.set_ylabel('Pixel Value')
        ax2.set_title('Vertical Line Profile')

        # Plot the first derivative of the line profile
        ax3.plot(range(height), line_first_derivative)
        ax3.set_xlabel('Y Axis (Height of Image)')
        ax3.set_ylabel('First Derivative')
        ax3.set_title('First Derivative of Vertical Line Profile')

        plt.tight_layout()
        plt.show()
    return line_values

def laplacian_edge_detection(image):
    # Read the image
    if image is None:
        print("Error loading image")
        return
    gray = image.copy()
    # Apply Laplacian edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Normalize the output to display as an image
    laplacian = np.uint8(np.absolute(laplacian))

    # Display original and Laplacian edge-detected images
    cv2.imshow('Original Image', image)
    cv2.imshow('Laplacian Edge Detection', laplacian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_contours(image):
    try:
        if image is None:
            print("Could not read the image. Please provide a valid image path.")
            return None

        # Check the number of image channels and convert to grayscale if necessary
        if len(image.shape) > 2 and image.shape[2] > 1:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Find contours
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        contour_image = np.copy(image)
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)  

        return contour_image

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def adaptive_threshold(image, block_size, constant):
    # Convert image to grayscale if it's not already in grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)
    
    return thresholded

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

def niblack_threshold(img, window_size=15, k=-0.2):
    # Convert image to grayscale if it's not already in grayscale
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Pad the image to handle borders when calculating local statistics
    padded_img = cv2.copyMakeBorder(img, window_size // 2, window_size // 2, window_size // 2, window_size // 2, cv2.BORDER_CONSTANT)

    thresholded_img = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Calculate local mean and standard deviation
            window = padded_img[i:i+window_size, j:j+window_size]
            mean, std_dev = np.mean(window), np.std(window)

            # Calculate threshold using Niblack's method
            threshold = mean + k * std_dev

            # Apply thresholding
            if img[i, j] > threshold:
                thresholded_img[i, j] = 255  # White (foreground)
            else:
                thresholded_img[i, j] = 0    # Black (background)

    return thresholded_img

def histogram_normalization(image):
    # Convert the image to grayscale if it's a color image
    gray = image

    # Calculate the histogram of the image
    hist, _ = np.histogram(gray.flatten(), 256, [0, 256])

    # Calculate the cumulative distribution function (CDF) of the histogram
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Perform histogram normalization
    normalized_image = np.interp(gray.flatten(), range(256), cdf_normalized).reshape(gray.shape)

    # # Display original and normalized images side by side
    # plt.figure(figsize=(8, 4))

    # plt.subplot(1, 2, 1)
    # plt.imshow(gray, cmap='gray')
    # plt.title('Original Image')
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(normalized_image, cmap='gray')
    # plt.title('Normalized Image')
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()

    return normalized_image

def adaptive_histogram_equalization(image, clip_limit=2.0, grid_size=(8, 8)):
    # Convert the image to grayscale if it's a color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Adaptive Histogram Equalization (AHE)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    ahe_image = clahe.apply(gray)

    # # Display original and AHE-enhanced images side by side
    # plt.figure(figsize=(8, 4))

    # plt.subplot(1, 2, 1)
    # plt.imshow(gray, cmap='gray')
    # plt.title('Original Image')
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(ahe_image, cmap='gray')
    # plt.title('Adaptive HE Image')
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()

    return ahe_image

def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    # Convert the image to grayscale if it's a color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Check if the image is already in the expected format (8-bit unsigned)
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    clahe_image = clahe.apply(gray)

    # Display original and CLAHE-enhanced images side by side
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(clahe_image, cmap='gray')
    plt.title('CLAHE Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return clahe_image

# go up from bottom of contamination and when you find big jump in pixel value, that is the end of contamination and do this in each direction for 30 pixels and then find the mode of these 3 values
def Get_Contamination_Height(image, middleOfContamination, bottomOfContamination, maxY):
    top = -1
    arrayWithPotentialTops = []
    start_x = middleOfContamination - 20
    end_x = middleOfContamination + 20

    for x in range(start_x, end_x):
        for y in range(bottomOfContamination-10, 0, -1):
            pixel_value = image[y, x]
            if pixel_value > 150:
                arrayWithPotentialTops.insert(0, y)
                break

    if (arrayWithPotentialTops):
        top = mode(arrayWithPotentialTops)

    print("top of Contamination:", top)
    return top


def get_contamination_range(image, maxY):
    img = image.copy()

    # Get the height and width of the image
    height, width = img.shape

    # Initialize variables to store the y values for each direction
    center_x = width // 2  # Start from the center column
    left_x = center_x - 400  # 100 columns to the left
    right_x = center_x + 400  # 100 columns to the right

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

    print("Start of contamination:", left_end)
    print("End of contamination:", right_end)

    print("width:", right_end - left_end)
    middle = ((right_end - left_end) // 2) + left_end
    print("middle:", middle)
    return left_end, right_end, middle


# starts in the middle and goes from bottom of the image to the top and finds the first white pixel, and go like that once in direction right and once in direction left
def get_contamination_bottom_height(image, maxY):
    img = image.copy()

    # Get the height and width of the image
    height, width = img.shape

    # Initialize variables to store the y values for each direction
    center_x = width // 2  # Start from the center column
    center_y = height - 1  # Start from the bottom row
    left_x = center_x - 100  # 100 columns to the left
    right_x = center_x + 100  # 100 columns to the right

    # Function to find the most common y value while moving up
    def find_most_common_y(x_start, x_end):
        y_values = []

        for x in range(x_start, x_end):
            y = height - 1  # Start from the bottom row
            while y >= 0:
                pixel_value = img[y, x]
                if pixel_value > 150: 
                    y_values.append(y)
                    break
                y -= 1

        if y_values:
            # Calculate the mode (the value that appears most frequently) of the y values
            mode_y = max(set(y_values), key=y_values.count)
        else:
            mode_y = -1  

        return mode_y

    # Find the most common y value in the center column
    center_result = find_most_common_y(center_x, center_x + 1)

    # Find the most common y value while moving to the right
    right_result = find_most_common_y(center_x, right_x + 1)

    # Find the most common y value while moving to the left
    left_result = find_most_common_y(left_x, center_x + 1)

    # find mode of these 3
    modeY = mode([center_result, right_result, left_result])
    print("modeY:", modeY)
    return modeY


def find_and_draw_contours(image, PreImage):
    # Find the contours
    contours, _ = cv2.findContours(
        image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours
    image_copy = image.copy()
    cv2.drawContours(image_copy, contours, -1, (255, 0, 255), 2)

    return image_copy


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
    """
    Applies median blur filter to the given image using the specified kernel size.

    Args:
    - image: Input image (numpy array).
    - kernel_size: Size of the kernel for the median blur filter.

    Returns:
    - Blurred image (numpy array).
    """

    # Apply median blur filter
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


def GetBottomLineOfContamination(image):
    height = 0
    # get first white occurence from the bottom left
    for row in range(image.shape[0] - 1, 0, -1):
        for col in range(image.shape[1]):
            if (image[row, col] > 100):
                height = row
                break
        if (height != 0):
            break
    print("height:", height)
    return height


def save_image(image, file_path, format="PNG"):
    try:
        if isinstance(image, np.ndarray):
            # Check if the specified format is supported by OpenCV
            valid_formats = ['PNG', 'JPEG', 'JPG', 'TIFF']
            if format.upper() in valid_formats:
                cv2.imwrite(file_path, image)
                print(f"Image saved to {file_path}")
            else:
                print(
                    f"Error saving the image: Unsupported image format '{format}'")
        else:
            print("Error saving the image: Invalid image format")
    except Exception as e:
        print(f"Error saving the image: {e}")


def remove_background_above_Tin_Ball(image, threshold=260, num_pixels_behind=5, num_pixels_ahead=5):
    # Get the dimensions of the image
    height, width = image.shape

    # Create a copy of the image to store the result
    result_image = image.copy()
    edge_values = []
    found = False
    # Iterate through each column
    for col in range(width):
        # Iterate through each row from bottom to top
        for row in range(height - num_pixels_ahead, num_pixels_ahead, -1):
            # Calculate the sum of the pixel values above the current pixel
            sum_above = int(np.sum(
                image[max(0, row - num_pixels_behind):row, col]))
            # Calculate the sum of the pixel values below the current pixel
            sum_below = int(np.sum(
                image[row + 1:min(height, row + num_pixels_ahead + 1), col]))
            # Check if the sum of the pixel values above and below the current pixel is greater than the threshold
            if (int(sum_below) - int(sum_above)) > threshold:
                # Set all pixels below the current pixel to zero
                # print("row:", row, "col:", col, "value:",
                #       abs(sum_below - sum_above))
                edge_values += [row]
                found = True
                result_image[:row + 1, col] = 0
                break

        if (found):
            found = False
            continue
        else:
            edge_values += [0]

    # print(edge_values)
    return result_image, edge_values


def find_start_end_indices(data, window_size=10, threshold=200):
    window_sum = []
    potential_start_indices = []
    potential_end_indices = []

    # Iterate through the data
    for index in range(len(data) - window_size):
        # get the sum of substraction window
        window_sum_x = np.sum(data[index:index + window_size])
        window_sum.append(window_sum_x)

    # go through the window sum and find the start and end indice, it should start when there will be a bigger jump from the previous value
    # and it should end when there will be a bigger jump from the next value
    for index in range(len(window_sum) - 1):
        if (window_sum[index] - window_sum[index + 1] > threshold):
            potential_start_indices.append(index)

    # find end but go from the end of the list
    for index in range(len(window_sum) - 1, 0, -1):
        if (window_sum[index] - window_sum[index - 1] > threshold):
            potential_end_indices.append(index)

    return potential_start_indices, potential_end_indices


def Take_10_pixels_In_Each_Row_And_Put_Them_In_Csv(image):
    height, width = image.shape[:2]

    window_size = 10  
    mat = np.zeros((height, width - window_size + 1), dtype=np.uint16)

    for row in range(height):
        for col in range(width - window_size + 1):
            # Extract a window of 10 pixels
            window = image[row, col:col + window_size]

            # Calculate the sum of the pixel values in the window
            pixel_sum = np.sum(window)
            if (pixel_sum > 1500):
                mat[row, col] = pixel_sum

    # Convert the 'mat' array to a NumPy array
    mat_as_np_array = np.array(mat)

    # Save the NumPy array to a CSV file
    np.savetxt('output.csv', mat_as_np_array, delimiter=',', fmt='%.2f')


def find_points(image, threshold=80):
    # Create an empty matrix to store the points
    mat = np.zeros_like(image)

    # Get height and width for grayscale or color images
    height, width = image.shape[:2]
    print(height, width)
    # Iterate through the columns from left to right
    for col in range(width):
        # Iterate through the rows from bottom to top
        for row in range(height - 1, 0, -1):  # Start from height - 1
            pixel_value = image[row, col]

            # Check if the row index is within bounds
            if row - 1 >= 0:
                subPixelValue = int(image[row - 1, col]) - int(pixel_value)
                if abs(subPixelValue) > threshold:

                    mat[row, col] = 255
                    break

    return mat


def load_images_from_folder(folder_path):
    image_paths = []
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif',
                        '*.bmp', "*.tif"] # Supported image extensions

    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(
            folder_path, '**', extension), recursive=True))

    return image_paths


def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# cutt off the edges of the image


def preprocess_image(image):
    return image[:-200, :]
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

    log_edges = (log_edges - np.min(log_edges)) / \
        (np.max(log_edges) - np.min(log_edges)) * 255

    # Convert to 8-bit unsigned integer (0-255)
    log_edges = np.uint8(log_edges)

    return log_edges


def otsu_thresholding(image):
    # Apply Otsu's thresholding
    _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresholded


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
    gradient_magnitude_image = gradient_magnitude(
        gradient_x_image, gradient_y_image)
    suppressed_image = non_maximum_suppression(
        gradient_magnitude_image, gradient_x_image, gradient_y_image)
    strong_edges, weak_edges = hysteresis_threshold(
        suppressed_image, low_threshold, high_threshold)
    edge_image = edge_tracking_by_hysteresis(strong_edges, weak_edges)
    return edge_image


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


def Get_Image_With_Most_frequent_Pixel_Red(image_path):
    # Load the image in grayscale
    image = load_image(image_path)
    image = preprocess_image(image)

    # Calculate the histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Find the mode (pixel value with the highest frequency)
    mode_pixel_value = np.argmax(hist)

    # Create a blank color image
    color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Set pixels with the mode value to red
    color_image[image == mode_pixel_value] = [255, 255, 255]  # White color

    plt.figure(figsize=(12, 6))

    # Plot the source grayscale image on the left
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title('Source Grayscale Image')

    # Display the color image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Color Image with Most Frequent Pixel in Red')
    plt.show()


def blurring(img):
    return apply_gaussian_blur(img)


def apply_gaussian_blur(image, kernel_size=(5, 5), sigma_x=0):
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma_x)
    return blurred_image


def apply_median_blur(image, kernel_size=5):
    blurred_image = cv2.medianBlur(image, kernel_size)
    return blurred_image


# this One looks like working the best
def apply_bilateral_filter(image, d=9, simgaColor=75, sigmaSpace=75):
    blurred_image = cv2.bilateralFilter(image, d, simgaColor, sigmaSpace)
    return blurred_image


def apply_fastNlMeansDenoisingColored(image):
    pass


def thresholding(img, thresh=127, maxval=255, type=cv2.THRESH_BINARY):
    ret, thresholding = cv2.threshold(img, thresh, maxval, type)
    return thresholding


def adaptive_threshold(image, block_size=31, c=-10):
    # Apply adaptive thresholding
    thresholded = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
    return thresholded


def parallel_shift_denoise(image, shift_direction=(1, 1)):
    shifted_image = np.roll(image, shift_direction, axis=(0, 1))
    denoised_image = (image + shifted_image) / 2
    return denoised_image


def post_process_contamination(detected_contamination):
    pass