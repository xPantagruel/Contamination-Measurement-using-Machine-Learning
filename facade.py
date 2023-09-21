from matplotlib.pyplot import imshow
from strategy import *
from observer import *
from image_processing import *
from helper_class import *
import urllib.request

class ContaminationMeasurementFacade:
    def __init__(self, strategy):
        self.strategy = strategy
        self.observers = []

    def attach_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self):
        for observer in self.observers:
            observer.update(self)

    def measure_contamination(self, image_path):
        image = load_image(image_path)
        preprocessed_image = preprocess_image(image)
        
        blurred_Image = blurring(preprocessed_image)
        threshold_Image = thresholding(blurred_Image)
        canny_image = canny_edge_detection(threshold_Image, 45,50, 7,2)
        
        # Display the images
        plt.figure(figsize=(12, 6))

        # Original Image
        plt.subplot(231)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        # Preprocessed Image
        plt.subplot(232)
        plt.imshow(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
        plt.title('Preprocessed Image')

        # Blurred Image
        plt.subplot(233)
        plt.imshow(cv2.cvtColor(blurred_Image, cv2.COLOR_BGR2RGB))
        plt.title('Blurred Image')

        # Thresholded Image
        plt.subplot(234)
        plt.imshow(cv2.cvtColor(threshold_Image, cv2.COLOR_BGR2RGB), cmap='gray')
        plt.title('Thresholded Image')

        # Canny Edge Detected Image
        plt.subplot(235)
        plt.imshow(cv2.cvtColor(canny_image, cv2.COLOR_BGR2RGB), cmap='gray')
        plt.title('Canny Edge Detected Image')

        plt.tight_layout()
        plt.show()
        
        detected_contamination = self.strategy.detect_contamination(preprocessed_image)
        refined_contamination = post_process_contamination(detected_contamination)
        self.notify_observers()
