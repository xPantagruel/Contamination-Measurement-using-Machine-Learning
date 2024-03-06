import os
import pandas as pd
from colorama import Fore, Style  # Import colorama modules for text coloring
import shutil  # Import shutil for file operations
import numpy as np
store_failed_images = True

def test_csv_data(processed_data):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    # folder_path = os.path.join(current_directory, "images")
    csv_file = os.path.join(current_directory, "contamination_measurements.csv")
    data = pd.read_csv(csv_file)

    succesed = 0
    failed = 0
    # Loop through processed data and compare with CSV
    for processed_item in processed_data:
        if processed_item is None:
            continue
        image_name = processed_item.ImageName
        csv_row = data[data['ImageName'] == image_name]

        if not csv_row.empty:
            csv_values = csv_row.iloc[0]

            bottom_height_diff = abs(processed_item.BottomHeightY - csv_values['BottomHeightY'])
            top_height_diff = abs(processed_item.TopHeightY - csv_values['TopHeightY'])
            if processed_item.ContaminationHeight < 0:
                contamination_height_diff = abs(csv_values['ContaminationHeight'] + processed_item.ContaminationHeight)
            else:
                contamination_height_diff = abs(processed_item.ContaminationHeight - csv_values['ContaminationHeight'])
            # contamination_height_diff = np.abs(np.abs(int(processed_item.ContaminationHeight)) - np.abs(int(csv_values['ContaminationHeight'])))

            similarity_threshold = 25

            if (
                bottom_height_diff <= similarity_threshold
                and top_height_diff <= similarity_threshold and 
                contamination_height_diff <= similarity_threshold
            ):
                succesed += 1
                print(Fore.GREEN + f"Values for {image_name} are close to the CSV values." + Style.RESET_ALL)
            else:
                failed += 1
                print (f"BottomHeightY: {processed_item.BottomHeightY} vs {csv_values['BottomHeightY']}")
                print (f"TopHeightY: {processed_item.TopHeightY} vs {csv_values['TopHeightY']}")
                print (f"ContaminationHeight: {processed_item.ContaminationHeight} vs {csv_values['ContaminationHeight']}")
                print (f"BottomHeightY diff: {bottom_height_diff}")
                print (f"TopHeightY diff: {top_height_diff}")
                print (f"ContaminationHeight diff: {contamination_height_diff}")

                if store_failed_images:
                    # Store the image in the folder for failed images
                    folder_with_images = r"C:\Users\matej.macek\OneDrive - Thermo Fisher Scientific\Desktop\BC Contamination Measurement\BC- FORK\ContaminationMeasurement\WholeDataset"
                    folder_for_failed_images = os.path.join(current_directory, "FailedImages")
                    image_path = os.path.join(folder_with_images, image_name)  # Assuming image is in this folder

                    # Create the directory if it doesn't exist
                    if not os.path.exists(folder_for_failed_images):
                        os.makedirs(folder_for_failed_images)

                    # Copy the image to the failed images folder
                    shutil.copy(image_path, folder_for_failed_images)
                
                print(Fore.RED + f"Values for {image_name} are not close to the CSV values. Image copied to failed images folder." + Style.RESET_ALL)
        else:
            print(f"No data found in CSV for {image_name}")

    print(f"Test results: {succesed} succesed, {failed} failed")
