# USER GUIDE FOR CONTAMINATION AUTOMATIC MEASUREMENT TOOLS

## General information
This repository contains implementations for a bachelor's thesis: Measuring the thickness of contamination layers in scanning electron microscopy using image processing created by Matěj Macek.

## Requirements
- Python 3.10.11 or higher
- Required libraries as specified in `requirements.txt` for each methods
- Access to the command line interface

## Installation
Before running the script, ensure all dependencies are installed:
        - pip install -r requirements.txt

## Usage edge Detection-Based Contamination Analyzer (EDCA)
For use of EDCA you have to go inside the folder "EDCA" in terminal.
### Basic Command
To run the script in the default mode without additional features:
        
        python main.py

### Optional Arguments
Enhance the script's functionality with the following optional arguments:

#### `--multiprocessing`
Use this flag to enable multiprocessing, which allows the script to process images concurrently across multiple CPU cores, thus speeding up the overall processing time.

        python main.py --multiprocessing

#### `--tests`
Perform tests on the processed data to validate the results against predefined benchmarks or expected outcomes.

        python main.py --tests

#### `--debug`
Enable detailed logging of processing steps, useful for debugging and ensuring that the script functions as expected.

        python main.py --debug

#### `--nano`
Switch to nanoscale mode for precise measurements at the nano level, necessary for processing images that require high accuracy in tiny dimensions.

        python main.py --nano

#### `--plot`
Plot images during processing for visual analysis or verification. This is particularly useful when visual feedback is required to understand how images are being processed.

        python main.py --plot

### Combining Flags
You can combine multiple flags to tailor the script’s execution to your needs:

        python main.py --multiprocessing --tests --debug --plot


## Example Use Cases
- **To perform a detailed test run across all datasets in folder Data_Sotrage\Error_Measurements_Datasets**:

        python main.py --multiprocessing --tests --multiprocessing

- **To perform a detailed test run in nanoscale mode**:

        python main.py --nano --tests --multiprocessing

## Usage DeepLabv3-based Contamination Layer Segmentation
For use of finetuning you have to go inside the folder DeeplabV3-based method in terminal.

## Installation
Before running the script, ensure all dependencies are installed (to be noted we were developing using wsl ubuntu):
        - pip install -r requirements.txt

### USAGE
For the finetuning use this command in terminal: 
        
        python FinetuningDeeplab.py 

This command will start finetuning. When you want to change the batch size or number of epochs you will have to edit it in the code where in the beggining there are global variables for it.
If you want to change the datasets its trained on, edit the folder Wholedataset in the same folder as the script.

For the usage of the notebook you will have to set the right current_directory path at the begining of the notebooks.

# Folder Content Tree
``` 
.
├───Data_Storage
│   ├───Augmented_Dataset
│   │   ├───Images
│   │   └───Masks
│   ├───Error_Measurements_Datasets             - Folder contains all datasets created for error measurements and finetuning
│   │   ├───Contamination_Only_Dataset
│   │   ├───Default_Dataset
│   │   ├───Uniq_Contamination_Only_Dataset
│   │   └───Uniq_Dataset
│   ├───Masks
│   │   ├───masks_wholedataset
│   │   └───Unique_Images_Masks
│   ├───Mask_ThresholdFinder_Dataset            - Images that the threshold finder utilizes to determine the appropriate threshold for probability maps.
│   ├───Nano_Measurement_Datasets               - Folder with images for measurements of error in nanometres.
│   │   ├───DefaultDatasetBeforeResize
│   │   ├───MasksBeforeResize
│   │   └───Uniq_Images
│   └───Original_Images
│   
├───DeeplabV3-based method
│   │   AutomatedTests.ipynb                    - Python notebook for automated testing of the model performance.
│   │   Error_Measurements_Results.csv          - Results from testing the model performance using errors metrics.
│   │   FinetuningDeeplab.py                    - Finetuning script.
│   │   GetMostAccurateMaskThreshold.py         - Script for acquire the best mask from probability mask output of model using the thresholding value.
│   │   HelperAnalysis.ipynb                    - Python notebook for looking at the results of the models outputs.
│   │   requirements.txt                        - Lists the libraries required for the scripts.
│   │
│   ├───csvWeightsAugmented                     - CSVs with values from mask threshold finder for specific finetuned models.
│   │       error_measurements_0.0031.csv
│   │       error_measurements_0.0063.csv
│   │       error_measurements_0.0125.csv
│   │       error_measurements_0.0250.csv
│   │
│   ├───csvWeightsContaminationOnly
│   │       error_measurements_0.0031.csv
│   │       error_measurements_0.0063.csv
│   │       error_measurements_0.0125.csv
│   │       error_measurements_0.0250.csv
│   │       error_measurements_Threshold_step_0.025.csv
│   │
│   ├───csvWeightsWithNoContaminationTraining
│   │       error_measurements_0.0031.csv
│   │       error_measurements_0.0063.csv
│   │       error_measurements_0.0125.csv
│   │       error_measurements_0.0250.csv
│   │
│   ├───TrainingResults                         - Stored csv from training with "log" files. Files started with "weights" are stored weights from finetuning process for each dataset.
│   │       logAugmented50Epochs.csv
│   │       logWeightsContaminationOnly.csv
│   │       weightsAugmented.pt
│   │       WeightsContaminationOnly.pt
│   │       weightsWithNoContaminationTraining.pt
│   │
│   └───WholeDataset
│       ├───Images
|       └───Masks
│
├───EDCA
│        AcquireDataset.py                       - This script processes images from a specified folder by cropping them and removing dimensions covered by masks.
│        AugmentateDataset.py                    - Generates augmented images from a specified folder to aid in model fine-tuning.
│        ContaminationMeasurementClass.py        - Provides the functionality to measure the height of contamination layers.
│        CreateTestCsvFromMasks.py               - Extracts masks from a designated folder, measures their top and bottom Y values and heights, and compiles these measurements along with their filenames into a CSV file.
│        DataClass.py                            - Defines a data class to clearly represent the measured values for each contamination layer, enhancing code clarity.
│        image_processing.py                     - Contains image processing functions used throughout the scripts, particularly supporting contamination height measurement.
│        main.py                                 - Main program handler that loads images from specific folders, measures their heights, and compares these measurements to predefined test values.
│        requirements.txt                        - Lists the libraries required for the scripts.
│        tests.py                                - Compares measured values against ground truth data, calculates errors, and outputs the results.
│
├─── contamination_measurements.csv                     - Csv folder with ground truth values for each image from the annotated mask.
├─── contamination_measurements_before_resized.csv      - Csv folder with ground truth values for each image before resize from the annotated mask.
├─── pixelWidth.csv                                     - The pixel width value in nanometers that should be applied as a multiplier to each pixel in the results for each image to achieve accurate scale values in nanometers.
├─── BC_text.pdf                                        - Pdf with text of the thesis.
├─── BC-thesis latex source code.zip                    - All latex source codes for thesis pdf generation.
├─── Poster.pdf                                         - Poster used at Excel at FIT.
└─── Video.mp4                                          - In case you can not play the video localy, you can use this url to youtube, where you can watch it online : https://youtu.be/9VaMpG_95_w
```

## Contact
For more assistance or to report bugs, please contact:
- Matěj Macek (xmacek27@stud.fit.vutbr.cz)