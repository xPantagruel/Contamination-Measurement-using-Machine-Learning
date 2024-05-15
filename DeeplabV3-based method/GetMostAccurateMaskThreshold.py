import os
import pandas as pd
from colorama import Fore, Style  
import shutil  
import numpy as np
import cv2 
import torch
import matplotlib.pyplot as plt
import pandas as pd
import time
from dataclasses import dataclass

store_failed_images = False
model_name = 'WeightsContaminationOnly'

@dataclass
class ProcessedData:
    ImageName: str
    BottomHeightY: int
    TopHeightY: int
    ContaminationHeight: int
    

# define the variable to store mean of error for height, top and bottom and also actual threshold value for mask
ERRORDICT = {'height': [], 'top': [], 'bottom': [], 'threshold': []}
MIN_THRESHOLD = 0.00
MAX_THRESHOLD = 1.00
MASK_THRESHOLD = 0.00
THRESHOLD_STEP = 0.01

def test_csv_data(processed_data):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    parent_directory = os.path.dirname(current_directory)
    
    csv_file = os.path.join(parent_directory, "contamination_measurements.csv")

    data = pd.read_csv(csv_file)

    succesed = 0
    failed = 0
    # for error measurement
    errorDict = {'height': [], 'top': [], 'bottom': []}

    # Loop through processed data and compare with CSV
    for processed_item in processed_data:
        if processed_item is None:
            continue
        image_name = processed_item.ImageName
        # change the image end from .png to .jpg
        image_name = image_name.replace(".jpg",".png")
        csv_row = data[data['ImageName'] == image_name]

        if not csv_row.empty:
            csv_values = csv_row.iloc[0]

            bottom_height_diff = abs(processed_item.BottomHeightY - csv_values['BottomHeightY'])
            top_height_diff = abs(processed_item.TopHeightY - csv_values['TopHeightY'])
            if processed_item.ContaminationHeight < 0:
                contamination_height_diff = abs(csv_values['ContaminationHeight'] + processed_item.ContaminationHeight)
            else:
                contamination_height_diff = abs(processed_item.ContaminationHeight - csv_values['ContaminationHeight'])

            similarity_threshold = 25
            # when the difference is less than the threshold, the test is considered successful 
            # this is because some contamination is really low so it is hard to measure it correctly and to say there is no contamination is correct
            if (processed_item.BottomHeightY == 0 or processed_item.TopHeightY == 0 ) and csv_values['ContaminationHeight'] < 40:
                print ("There is none or low contamination in the image. Skipping...")
                succesed += 1
                continue
            
            if (
                bottom_height_diff <= similarity_threshold
                and top_height_diff <= similarity_threshold and 
                contamination_height_diff <= similarity_threshold
            ):
                # calculate the ERROR 
                succesed += 1
                # FILL DICTIONARY ERROR MEASUREMENT ----------------------------------
                errorDict['height'].append(bottom_height_diff)
                errorDict['top'].append(top_height_diff)
                errorDict['bottom'].append(contamination_height_diff)
                # END OF FILLING DICT ERROR MEASUREMENT ---------------------------
            else:
                failed += 1

                if store_failed_images:
                    # Store the image in the folder for failed images
                    folder_with_images = os.path.join(current_directory, "Data_Storage", "WholeDataset")
                    folder_for_failed_images = os.path.join(current_directory, "FailedImages")
                    image_path = os.path.join(folder_with_images, image_name)  # Assuming image is in this folder

                    # Create the directory if it doesn't exist
                    if not os.path.exists(folder_for_failed_images):
                        os.makedirs(folder_for_failed_images)

                    # Copy the image to the failed images folder
                    shutil.copy(image_path, folder_for_failed_images)
        else:
            print(f"No data found in CSV for {image_name}")

    # CALCULATE THE ERROR MEASUREMENT ----------------------------------
    mae = {key: np.mean(values) for key, values in errorDict.items()}
    # END OF CALCULATING ERROR MEASUREMENT ---------------------------

    # store the error measurements
    ERRORDICT['height'].append(mae['height'])
    ERRORDICT['top'].append(mae['top'])
    ERRORDICT['bottom'].append(mae['bottom'])
    ERRORDICT['threshold'].append(MASK_THRESHOLD)
    
    print(f"Test results: {succesed} succesed, {failed} failed")

def measure_color_height(mask, color=255):
    # Find the indices of yellow (or specified color) pixels
    yellow_pixels = mask == color
    
    # If there are no yellow pixels, return None
    if not np.any(yellow_pixels):
        return None, None
    
    # Find the middle column
    middle_col = mask.shape[1] // 2
    
    # Find indices of yellow pixels in the middle column
    middle_col_indices = np.nonzero(yellow_pixels[:, middle_col])
    
    if len(middle_col_indices[0]) == 0:
        return None, None  # No yellow pixels found
    
    # Get the top and bottom heights
    top_height = middle_col_indices[0].min()
    bottom_height = middle_col_indices[0].max()
    
    return bottom_height, top_height

start_time = time.time()

# Load the trained model and set it to evaluation mode
model = torch.load('./TrainingResults/' + model_name + '.pt')
model.eval()

current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
# Directory containing images
image_dir = os.path.join(parent_directory, "Data_Storage", "Mask_ThresholdFinder_Dataset")

# Initialize a list to store threshold values
threshold_values = []
THRESHOLD_STEP = 0.025
MAX_THRESHOLD = 1.00
MIN_THRESHOLD = 0.00
DONE = False

while not DONE:
    while MASK_THRESHOLD < MAX_THRESHOLD: 
        MASK_THRESHOLD += THRESHOLD_STEP
        threshold_values.append(MASK_THRESHOLD)
        print(f"Mask threshold: {MASK_THRESHOLD}")
        # Process each image in the directory
        i = 670
        processed_data = []

        for filename in os.listdir(image_dir):
            if i == 0:
                break
            i -= 1

            # Construct paths to the image
            img_path = os.path.join(image_dir, filename)
            # Read the image
            img = cv2.imread(img_path)

            # Resize the image to fit the model's expected input size and prepare it for the model
            resized_img = cv2.resize(img, (480, 320))
            img_for_model = resized_img.transpose(2, 0, 1).reshape(1, 3, 320, 480)
            img_tensor = torch.from_numpy(img_for_model).type(torch.cuda.FloatTensor) / 255

            # Apply the model to the image
            with torch.no_grad():
                a = model(img_tensor)

            # Get the probability map
            probability_map = a['out'].cpu().detach().numpy()[0][0]

            # Get the predicted mask
            predicted_mask = probability_map > MASK_THRESHOLD

            # Convert the mask to uint8
            predicted_mask_uint8 = predicted_mask.astype(np.uint8) * 255

            # Resize the mask to match the resized image
            resized_mask = cv2.resize(predicted_mask_uint8, (resized_img.shape[1], resized_img.shape[0]))

            # Convert the mask to 3 channels to blend with the original image
            resized_mask_3_channels = cv2.merge((resized_mask, resized_mask, resized_mask))

            # Blend the mask with the original image
            overlay = cv2.addWeighted(resized_img, 0.7, resized_mask_3_channels, 0.3, 0)
            # resize the overlay to the original image size 1024x768
            overlay = cv2.resize(overlay, (img.shape[1], img.shape[0]))

            # resize the mask to the original image size 1024x768
            mask = cv2.resize(resized_mask, (img.shape[1], img.shape[0]))


            top_height, bottom_height = measure_color_height(mask, color=255)

            if top_height is None or bottom_height is None:
                processed_data.append(ProcessedData(ImageName=filename, BottomHeightY=0, TopHeightY=0, ContaminationHeight=0))
                continue
            
            processed_data_instance = ProcessedData(ImageName=filename, BottomHeightY=int(bottom_height), TopHeightY=int(top_height), ContaminationHeight=int(bottom_height - top_height))
            processed_data.append(processed_data_instance)

        test_csv_data(processed_data)
        end_time = time.time()
        execution_time = end_time - start_time
        print("Total execution time:", execution_time, "seconds")
        
    # store values in csv 
    df = pd.DataFrame(ERRORDICT)
    file_name = f'error_measurements_{THRESHOLD_STEP:.4f}.csv'
    csv_directory = "csv" + model_name
    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)
    file_path = os.path.join(csv_directory, file_name)
    try:
        df.to_csv(file_path)
        print("CSV file has been saved successfully to", file_path)
    except Exception as e:
        print("Failed to write CSV file:", e)
        
    # find in the dictionary the lowest value sum for the mask threshold top and bottom
    min_sum = 999999
    for i in range(len(ERRORDICT['height'])):
        sum = ERRORDICT['top'][i] + ERRORDICT['bottom'][i]
        if sum < min_sum:
            min_sum = sum
            min_index = i

    if THRESHOLD_STEP <= 0.005:
        DONE = True
    else :
        print("Min index:", min_index)
        # find new step or save values 
        min_threshold = ERRORDICT['threshold'][min_index]
        THRESHOLD_STEP /= 2  # Reduce step size
        MIN_THRESHOLD = min_threshold - THRESHOLD_STEP * 5  # Recalculate around the current mask threshold
        MAX_THRESHOLD = min_threshold + THRESHOLD_STEP * 5
        MASK_THRESHOLD = MIN_THRESHOLD  # Reset to the lower bound for the next set of iterations

        threshold_values = []
        ERRORDICT = {'height': [], 'top': [], 'bottom': [], 'threshold': []}
        print("New step:", THRESHOLD_STEP)
        print("New mask threshold:", MASK_THRESHOLD)

plt.plot(threshold_values, ERRORDICT['height'], label='Height')
plt.plot(threshold_values, ERRORDICT['top'], label='Top')
plt.plot(threshold_values, ERRORDICT['bottom'], label='Bottom')
plt.xlabel('Threshold')
plt.ylabel('Mean Absolute Error')
plt.title('Error measurements')
plt.legend()
plt.show()

# find the lowest value for the mask threshold top and bottom
min_sum = 999999
for i in range(len(ERRORDICT['height'])):
    sum = ERRORDICT['top'][i] + ERRORDICT['bottom'][i]
    if sum < min_sum:
        min_sum = sum
        min_index = i


print("Min threshold value:", ERRORDICT['threshold'][min_index])