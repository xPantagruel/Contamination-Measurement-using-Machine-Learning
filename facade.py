from matplotlib.pyplot import imshow
from strategy import *
from observer import *
from image_processing import *
from helper_class import *
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
        # Display the Canny edges image
        cv2.imshow('Canny Edges', canny_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # create_imagelist_edge_detection(preprocessed_image)
        
        detected_contamination = self.strategy.detect_contamination(preprocessed_image)
        refined_contamination = post_process_contamination(detected_contamination)
        self.notify_observers()
