import cv2
import numpy as np
import os
import csv

def measure_red_width(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range of red color in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    
    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # Check if there are any red pixels in the image
    if np.max(mask) == 0:
        print(f"No red pixels found in {image_path}")
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
    
    # Draw the bounding rectangle on the original image
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Get the most left and most right points
    most_left = (x, y + h // 2)
    most_right = (x + w, y + h // 2)
    
    # Measure width of the red area
    red_width = w
    
    # # Show the result
    # cv2.imshow('Result', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return red_width, most_left, most_right

def measure_red_height(image_path, column):
    print (f"Measuring red height in column {column} of {image_path}")
    # go through the column and find the first and last red pixel
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    first_red = None
    last_red = None
    for i in range(mask.shape[0]):
        if mask[i, column] == 255:
            first_red = (column, i)
            break
    for i in range(mask.shape[0] - 1, -1, -1):
        if mask[i, column] == 255:
            last_red = (column, i)
            break
    if first_red is None or last_red is None:
        print(f"No red pixels found in column {column} of {image_path}")
        return None

    height = last_red[1] - first_red[1]
    
    # show in the image the first and last red pixel with lines 
    
    # cv2.line(img, first_red, (first_red[0] + 10, first_red[1]), (122, 122, 122), 2)
    # cv2.line(img, last_red, (last_red[0] + 10, last_red[1]), (0, 255, 0), 2)
    # cv2.imshow('Result', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # last red  = top 
    # first red = bottom    
    return height,  first_red[1],last_red[1]

# Path to the folder containing images
# in actual folder will be folder maskks with images
current_directory = os.path.dirname(os.path.realpath(__file__))
folder_path = os.path.join(current_directory, "masks")

# CSV file path
csv_file_path = os.path.join(current_directory, "contamination_measurements.csv")

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
            red_width, most_left, most_right = measure_red_width(image_path)
            if red_width is not None:
                starting_column = most_left[0] + red_width // 2
                height,bottom_height_y,top_height_y = measure_red_height(image_path, starting_column)
                print(f"Image: {filename}, Red Width: {red_width}, Most Left Point: {most_left}, Most Right Point: {most_right}, Height: {height}")
                
            if most_left is not None:
                writer.writerow({'ImageName': filename, 'BottomHeightY': bottom_height_y, 'TopHeightY': top_height_y, 'ContaminationHeight': height})    
            else:
                writer.writerow({'ImageName': filename, 'BottomHeightY': 0, 'TopHeightY': 0, 'ContaminationHeight': 0})
