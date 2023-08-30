class ContaminationDetectionStrategy:
    def detect_contamination(self, image):
        pass

class ThresholdingStrategy(ContaminationDetectionStrategy):
    def detect_contamination(self, image):
        print("Ahoj\n")

class EdgeDetectionStrategy(ContaminationDetectionStrategy):
    def detect_contamination(self, image):
        pass

class MachineLearningStrategy(ContaminationDetectionStrategy):
    def detect_contamination(self, image):
        pass
