# @file main.py
# @brief main program handler where are loaded image from specific folders, measure their height and compare the result to test values
# @author Matěj Macek (xmacek27@fit.vutbr.cz)
# @date 4.5.2024
 
import os
from ContaminationMeasurementClass import ContaminationMeasurementClass
from image_processing import load_images_from_folder
import multiprocessing
from DataClass import ProcessedData 
from tests import test_csv_data
from PIL import Image
import csv

use_multiprocessing = False
do_Tests = True
DEBUG = False
nano = True

# foldername,mae top,mse top ,rmse top ,median top,mae bottom,mse bottom,rmse bottom,median bottom,mae height,mse height,rmse height,median height
Error_Measurements_Result = {'foldername': [],'succesed':0,'failed':0,'Zero_Contamination_Badly_Measured':0, 'mae top': [], 'mse top': [], 'rmse top': [], 'median top': [], 'mae bottom': [], 'mse bottom': [], 'rmse bottom': [], 'median bottom': [], 'mae height': [], 'mse height': [], 'rmse height': [], 'median height': []}
Error_Measurements_Results = []
def process_image(image_path):
    if DEBUG:
        print("Processing image: " + image_path)

    try:
        contamination_measurement = ContaminationMeasurementClass()
        BottomHeight, TopHeight = contamination_measurement.measure_contamination(image_path,plot=True)

        if BottomHeight is None or TopHeight is None:
            raise ValueError(f"Received None for heights in image: {image_path}")

        Height = BottomHeight - TopHeight
        image_name = os.path.basename(image_path)
        data_instance = ProcessedData(ImageName=image_name, BottomHeightY=int(BottomHeight), TopHeightY=int(TopHeight), ContaminationHeight=int(Height))

        if DEBUG:
            print("Finished processing image: " + image_path)

        return data_instance
    except Exception as e:
        return None
    
def TestSpecificFolder(folder_path):
    image_paths = load_images_from_folder(folder_path)
    Results = []

    if use_multiprocessing:
        num_processes = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_processes-1) as pool:
            Results = pool.map(process_image, image_paths)
    else:
        for image_path in image_paths:
            data_instance = process_image(image_path)
            if data_instance is not None:
                Results.append(data_instance)

    if do_Tests:
        mae, mse, rmse, median, succesed, failed, Zero_Contamination_Badly_Measured = test_csv_data(Results,nanoscale_mode=nano)
        base_name = os.path.basename(folder_path)
        Error_Measurements_Result  = {'foldername': base_name,'succesed': succesed,'failed': failed,'Zero_Contamination_Badly_Measured':Zero_Contamination_Badly_Measured,'mae top': mae['top'], 'mse top': mse['top'], 'rmse top': rmse['top'], 'median top': median['top'], 'mae bottom': mae['bottom'], 'mse bottom': mse['bottom'], 'rmse bottom': rmse['bottom'], 'median bottom': median['bottom'], 'mae height': mae['height'], 'mse height': mse['height'], 'rmse height': rmse['height'], 'median height': median['height']}
        print(Error_Measurements_Result)
        Error_Measurements_Results.append(Error_Measurements_Result)

def TestErrorMeasurementAccrossAllDatasets(folder_paths):
    for folder_path in folder_paths:
        # print only the folder name   
        folder_path_name = os.path.basename(folder_path)
        print(f"Testing folder: {folder_path_name}")
        TestSpecificFolder(folder_path)
        print ("---------------------------------------------------------------------------")

    # store in the result error csv Error_Measurements_Results
    with open('Error_Measurements_Results.csv', mode='w') as file:
        writer = csv.DictWriter(file, fieldnames=Error_Measurements_Result.keys())
        writer.writeheader()
        for data in Error_Measurements_Results:
            writer.writerow(data)

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.realpath(__file__))
    if nano:
        # folder_path = os.path.join(current_directory, r"Data_Storage\BeforeResized_Datasets\Uniq_Images")
        # folder_path = os.path.join(current_directory, r"Data_Storage\BeforeResized_Datasets\ImagesBeforeResize")
        folder_path = os.path.join(current_directory, r"Data_Storage/UniqueImages")
        TestSpecificFolder(folder_path)
    else:
        folder_path = os.path.join(current_directory, "Data_Storage/Error_Measurements_Datasets/")
        folder_paths = []
        # in folder Error_Measurements_Datasets there are folders with images
        for folder in os.listdir(folder_path):
            folder_paths.append(os.path.join(folder_path, folder))

        TestErrorMeasurementAccrossAllDatasets(folder_paths)

    print("All processes have finished.")
