from matplotlib.pyplot import imshow
from image_processing import *
from helper_class import *

class ContaminationMeasurementClass:
    def measure_contamination(self, image_path):
        image = load_image(image_path)
        preprocessed_image = self.cutt_off_edges(image)
        blurred_image = self.blur_image(preprocessed_image)
        thresholded_image = self.threshold_image(blurred_image)
        canny_image = self.detect_edges(thresholded_image)
        self.visualize([image, preprocessed_image, blurred_image, thresholded_image, canny_image],
                       ["Original Image", "Preprocessed Image", "Blurred Image", "Thresholded Image", "Canny Edge Detected Image"])
    
    def cutt_off_edges(self, image):
        return preprocess_image(image)
    
    def blur_image(self, image):
        return blurring(image) 
    
    def threshold_image(self, image):
        return thresholding(image)
    
    def detect_edges(self, image):
        return canny_edge_detection(image, 45, 50, 7, 2)
    
    def visualize(self, image_list, title_list):
        # Display the images
        plt.figure(figsize=(12, 6))

        for i in range(5):
            plt.subplot(2, 3, i + 1)
            if len(image_list[i].shape) == 2:  # Grayscale image
                plt.imshow(image_list[i], cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(image_list[i], cv2.COLOR_BGR2RGB))
            plt.title(title_list[i])

        plt.tight_layout()
        plt.show()