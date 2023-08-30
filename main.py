from facade import ContaminationMeasurementFacade
from strategy import ThresholdingStrategy
from observer import AnalysisObserver, VisualizationObserver

if __name__ == "__main__":
    thresholding_strategy = ThresholdingStrategy()
    facade = ContaminationMeasurementFacade(thresholding_strategy)

    analysis_observer = AnalysisObserver()
    visualization_observer = VisualizationObserver()

    facade.attach_observer(analysis_observer)
    facade.attach_observer(visualization_observer)

    image_path = "path_to_your_image.jpg"
    facade.measure_contamination(image_path)
    print("AhojS.")
