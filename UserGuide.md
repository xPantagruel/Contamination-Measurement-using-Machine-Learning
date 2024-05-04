# USER GUIDE TO CONTAMINATION AUTOMATIC MEASURENT TOOL CALLED EDCA

firstly install all dependencies in requiremnts using this command:
        
        - Re

## Commnad for start the EDCA

        - python main.py

This will start the whole program and start tests with evaluation in the end at all datasets created.

By default the script will test whole images in folder and subfolders of \Data_Storage\Error_Measurements_Datasets and the results will be stored in csv "Error_Measurements_Results.csv". For the comparision is used "contamination_measurements.csv" where are ground truth data for each image.

To measure only nanometres tests you will have set in main.py boolean variable "nano" to true. CSV file pixelWidth.csv contains for each image number to multiply the result error number of pixels to convert it to nanometres. For this purpose it will use the "contamination_measurements_before_resized.csv" where are ground truth data for images before resized.

In case you want to see some plots from the data you enter main.py file, where is boolean variable called "plot", which if you set to true it will show you for each image in folder the results of crucial steps like starting point plot, top and bottom measurement plots of gradient line profile.

Each of the scripts has its own very brief manual what to do and what will be the output of the specific script.

# Scripts for measurement and csv files with ground truth values
- main.py
- ContaminationMeasurementClass.py
- DataClass.py
- tests.py
- image_processing.py
- pixelWidth.csv 
- contamination_measurements.csv
- contamination_measurements_before_resized.csv

# Tools used before measurement
- AcquireDataset.py
- AugmentateDataset.py
- CreateTestCsvFromMasks.py

# Storage with Images
- Data_Storage
    - Contamination_Only_Dataset 
    - Default_Dataset
    - Uniq_Contamination_Only_Dataset
    - Uniq_Dataset
- Nanoscale
    - Uniq_Images
    - ImagesBeforeResize
