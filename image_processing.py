import os
import glob
import cv2
from skimage import color, filters
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_laplace
from PIL import Image
from statistics import mode

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
                if pixel_value > 150:  # Adjust the threshold value as needed
                    y_values.append(y)
                    break
                y -= 1

        if y_values:
            # Calculate the mode (the value that appears most frequently) of the y values
            mode_y = max(set(y_values), key=y_values.count)
        else:
            mode_y = -1  # You can choose an appropriate default value

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
            if pixel_value > 150:  # Adjust the threshold value as needed
                white_pixel_heights.append(y)
                break

    if white_pixel_heights:
        # Calculate the mode (the value that appears most frequently) of the white pixel heights
        mode_height = max(set(white_pixel_heights),
                          key=white_pixel_heights.count)
    else:
        # If no white pixels are found, return a default value (you can change this as needed)
        mode_height = -1  # You can choose an appropriate default value

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

    window_size = 10  # You want to sum 10 consecutive pixels
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
    # Define a list to store image file paths
    image_paths = []
    # Use the 'glob' function to search for image files in the directory and its subdirectories
    # You can specify different image file extensions like '*.jpg', '*.png', '*.jpeg', etc.
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif',
                        '*.bmp', "*.tif"]  # Add more extensions as needed

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

    # Adjust the range of values for visualization
    log_edges = (log_edges - np.min(log_edges)) / \
        (np.max(log_edges) - np.min(log_edges)) * 255

    # Convert to 8-bit unsigned integer (0-255)
    log_edges = np.uint8(log_edges)

    return log_edges


def otsu_thresholding(image):
    # Convert the input image to grayscale if it's not already.
    if len(image.shape) == 3:
        image = color.rgb2gray(image)

    # Calculate the gradient magnitude using Sobel filters.
    gradient_x = np.abs(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3))
    gradient_y = np.abs(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3))
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Normalize gradient magnitude to [0, 255]
    gradient_magnitude = ((gradient_magnitude - np.min(gradient_magnitude)) /
                          (np.max(gradient_magnitude) - np.min(gradient_magnitude)) * 255).astype(np.uint8)

    # Calculate Otsu's threshold
    _, thresholded_image = cv2.threshold(
        gradient_magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresholded_image


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
    image = preprocess_image(image)
    # Calculate the histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(image.ravel(), 256, [0, 256])
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.show()


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
