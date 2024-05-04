# @file CreateTestCsvFromMasks.py
# @brief this python script from specific folder you define ("folder_path") takes masks and measure their height,bottom and top y values and store all in the csv file with their names
# @author Matěj Macek (xmacek27@fit.vutbr.cz)
# @date 4.5.2024

import cv2
import numpy as np
import os
import csv

# define the folder path to your masks
folder_path = r''

def measure_width(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range of red color in HSV
    lower = np.array([0, 100, 100])
    upper = np.array([10, 255, 255])
    
    # Threshold the HSV image to get only coloured colors
    mask = cv2.inRange(hsv, lower, upper)
    
    # Check if there are any coloured pixels in the image
    if np.max(mask) == 0:
        print(f"No coloured pixels found in {image_path}")
        return None, None, None
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"No contours found in {image_path}")
        return None, None, None
    
    # Find the contour with the maximum area
    max_contour = max(contours, key=cv2.contourArea)
    
    # Get the bounding rectangle of the maximum contour
    x, y, w, h = cv2.boundingRect(max_contour)
    
    # Get the most left and most right points
    most_left = (x, y + h // 2)
    most_right = (x + w, y + h // 2)
    
    # Measure width of the area
    red_width = w
    
    return red_width, most_left, most_right

def measure_height(image_path, column):
    print (f"Measuring height in column {column} of {image_path}")
    # go through the column and find the first and last coloured pixel
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 100, 100])
    upper = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    first = None
    last = None
    for i in range(mask.shape[0]):
        if mask[i, column] == 255:
            first = (column, i)
            break
    for i in range(mask.shape[0] - 1, -1, -1):
        if mask[i, column] == 255:
            last = (column, i)
            break
    if first is None or last is None:
        print(f"No coloured pixels found in column {column} of {image_path}")
        return None

    height = last[1] - first[1]
    
    return height,  first[1],last[1]

# Path to the folder containing images
# in actual folder will be folder maskks with images
current_directory = os.path.dirname(os.path.realpath(__file__))
folder_path = os.path.join(current_directory, "Data_Storage/maskResizedUniq/Unique_Images_Masks")
# CSV file path
csv_file_path = os.path.join(current_directory, "contamination_measurements_before_resized.csv")

# Create CSV file with columns: ImageName, BottomHeightY, TopHeightY, ContaminationHeight
with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = ['ImageName', 'BottomHeightY', 'TopHeightY', 'ContaminationHeight']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    height = None
    # Iterate through all images in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            width, most_left, most_right = measure_width(image_path)
            if width is not None:
                starting_column = most_left[0] + width // 2
                height,bottom_height_y,top_height_y = measure_height(image_path, starting_column)
                print(f"Image: {filename}, Width: {width}, Most Left Point: {most_left}, Most Right Point: {most_right}, Height: {height}")

            if most_left is not None:
                filename = filename.replace(".jpg", "")
                filename = filename.replace(".png", "")
                filename = filename.replace(".jpeg", "")
                filename = filename.replace(".tif", "")
                writer.writerow({'ImageName': filename, 'BottomHeightY': bottom_height_y, 'TopHeightY': top_height_y, 'ContaminationHeight': height})    
            else:
                filename = filename.replace(".jpg", "")
                filename = filename.replace(".png", "")
                filename = filename.replace(".jpeg", "")
                filename = filename.replace(".tif", "")
                writer.writerow({'ImageName': filename, 'BottomHeightY': 0, 'TopHeightY': 0, 'ContaminationHeight': 0})