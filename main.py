import os
from ContaminationMeasurementClass import ContaminationMeasurementClass
from image_processing import load_images_from_folder
import multiprocessing


def process_image(image_path):
    print("Processing image: " + image_path)
    contamination_measurement = ContaminationMeasurementClass()
    contamination_measurement.measure_contamination4(image_path)
    print("Finished processing image: " + image_path)


if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.realpath(__file__))
    folder_path = os.path.join(current_directory, "images/Default")

    image_paths = load_images_from_folder(folder_path)

    use_multiprocessing = False

    if use_multiprocessing:
        # Use the number of CPU cores as the number of processes
        num_processes = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_processes)

        pool.map(process_image, image_paths)
        pool.close()
        pool.join()
    else:
        # Run in single-process (normal) mode
        for image_path in image_paths:
            process_image(image_path)

    print("All processes have finished.")
