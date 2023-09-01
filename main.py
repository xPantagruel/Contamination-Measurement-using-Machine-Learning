import os
from facade import ContaminationMeasurementFacade
from strategy import ThresholdingStrategy
from observer import AnalysisObserver, VisualizationObserver
from image_processing import load_images_from_folder
import matplotlib.pyplot as plt
from PIL import Image

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
        
    print("Starting facade...")
    print("Facade will now measure contamination for all images in the folder")
    
    # Create a Matplotlib figure to display the images
    fig = plt.figure()

    # Loop through the image paths and display each image in grayscale
    for image_path in image_paths:
        # Open the image using Pillow
        image = Image.open(image_path)

        # Convert the image to grayscale (L mode)
        image = image.convert('L')

        # Display the grayscale image using Matplotlib with the 'gray' colormap
        plt.imshow(image, cmap='gray')

        # Add a title with the image file name
        plt.title(image_path)

        # Click to skip to the next image
        plt.waitforbuttonpress()

        # Clear the current figure to display the next image
        plt.clf()

    # Close the Matplotlib figure when done
    plt.close(fig)