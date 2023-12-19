import os
from ContaminationMeasurementClass import ContaminationMeasurementClass
from image_processing import load_images_from_folder
import multiprocessing
from DataClass import ProcessedData 
from tests import test_csv_data

use_multiprocessing = True
do_Tests = True

def process_image(image_path):
    print("Processing image: " + image_path)
    contamination_measurement = ContaminationMeasurementClass()
    BottomHeight,TopHeight = contamination_measurement.measure_contamination5(image_path)
    
    Height = BottomHeight - TopHeight
    image_name = os.path.basename(image_path)

    data_instance = ProcessedData(ImageName=image_name, BottomHeightY=BottomHeight, TopHeightY=TopHeight, ContaminationHeight=Height)
    print("Finished processing image: " + image_path)
    return data_instance


if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.realpath(__file__))
    folder_path = os.path.join(current_directory, "images")

    image_paths = load_images_from_folder(folder_path)

    Results  = []

    if use_multiprocessing:
        # Use the number of CPU cores as the number of processes
        num_processes = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_processes)

        pool.map(process_image, image_paths)

        Results = pool.map(process_image, image_paths)

        pool.close()
        pool.join()
    else:
        # Run in single-process (normal) mode
        for image_path in image_paths:
            data_instance = process_image(image_path)
            Results.append(data_instance)
        
    if do_Tests:
        test_csv_data(Results)
    
    print("Results: ", Results)

    print("All processes have finished.")
