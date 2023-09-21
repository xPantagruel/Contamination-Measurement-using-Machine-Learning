import os
from ContaminationMeasurementClass import ContaminationMeasurementClass
from image_processing import load_images_from_folder
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == "__main__":
    #use this directory and the folder images 
    current_directory = os.path.dirname(os.path.realpath(__file__))
    folder_path = os.path.join(current_directory, "images")
        
    # Load images
    image_paths = load_images_from_folder(folder_path)
    
    contamination_measurement = ContaminationMeasurementClass()

    print("Starting ...")
    for image_path in image_paths:
        print("Processing image: " + image_path)
        contamination_measurement.measure_contamination(image_path)
        
    print("Finished ...")