from strategy import *
from observer import *
from image_processing import *

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
        detected_contamination = self.strategy.detect_contamination(preprocessed_image)
        refined_contamination = post_process_contamination(detected_contamination)
        self.notify_observers()
