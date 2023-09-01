import os
from facade import ContaminationMeasurementFacade
from strategy import ThresholdingStrategy
from observer import AnalysisObserver, VisualizationObserver
from image_processing import load_images_from_folder
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #use this directory and the folder images 
    current_directory = os.path.dirname(os.path.realpath(__file__))
    folder_path = os.path.join(current_directory, "images")
    
    # Load images
    image_paths = load_images_from_folder(folder_path)
    
    thresholding_strategy = ThresholdingStrategy()
    facade = ContaminationMeasurementFacade(thresholding_strategy)

    analysis_observer = AnalysisObserver()
    visualization_observer = VisualizationObserver()

    facade.attach_observer(analysis_observer)
    facade.attach_observer(visualization_observer)

    for image_path in image_paths:
        print("Tu")
        # show the image using plot 
        plt.imshow(image_path)
        facade.measure_contamination(image_path)