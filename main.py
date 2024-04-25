import os
from ContaminationMeasurementClass import ContaminationMeasurementClass
from image_processing import load_images_from_folder
import multiprocessing
from DataClass import ProcessedData 
from tests import test_csv_data
from PIL import Image

use_multiprocessing = False
do_Tests = True
DEBUG = False

def process_image(image_path):
    if DEBUG:
        print("Processing image: " + image_path)

    try:
        contamination_measurement = ContaminationMeasurementClass()
        BottomHeight, TopHeight = contamination_measurement.measure_contamination(image_path)

        if BottomHeight is None or TopHeight is None:
            raise ValueError(f"Received None for heights in image: {image_path}")

        Height = BottomHeight - TopHeight
        image_name = os.path.basename(image_path)
        data_instance = ProcessedData(ImageName=image_name, BottomHeightY=int(BottomHeight), TopHeightY=int(TopHeight), ContaminationHeight=int(Height))

        if DEBUG:
            print("Finished processing image: " + image_path)

        return data_instance
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
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
        test_csv_data(Results)

def TestErrorMeasurementAccrossAllDatasets(folder_paths):
    for folder_path in folder_paths:
        # print only the folder name   
        folder_path_name = os.path.basename(folder_path)
        print(f"Testing folder: {folder_path_name}")
        TestSpecificFolder(folder_path)

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.realpath(__file__))
    folder_path = os.path.join(current_directory, "Data_Storage/Error_Measurements_Datasets/")
    folder_path_specific = os.path.join(current_directory, "Data_Storage/Images/fail")

    TestSpecificFolder(folder_path_specific)

    # folder_paths = []
    # # in folder Error_Measurements_Datasets there are folders with images
    # for folder in os.listdir(folder_path):
    #     folder_paths.append(os.path.join(folder_path, folder))

    # TestErrorMeasurementAccrossAllDatasets(folder_paths)

    print("All processes have finished.")
