import os
import pandas as pd
from colorama import Fore, Style  # Import colorama modules for text coloring
import shutil  # Import shutil for file operations
import numpy as np
import matplotlib.pyplot as plt

store_failed_images = False
DEBUG = False
def test_csv_data(processed_data):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    csv_file = os.path.join(current_directory, "contamination_measurements.csv")

    data = pd.read_csv(csv_file)

    succesed = 0
    failed = 0
    Zero_Contamination_Badly_Measured = 0
    # for error measurement
    errorDict = {'height': [], 'top': [], 'bottom': []}
    BaddlyValuatedNonContaminated = {'height': []}
    # Loop through processed data and compare with CSV
    for processed_item in processed_data:
        if processed_item is None:
            continue
        image_name = processed_item.ImageName
        # change the image end from .png to .jpg
        image_name = image_name.replace(".jpg",".png")
        csv_row = data[data['ImageName'] == image_name]

        if not csv_row.empty:
            csv_values = csv_row.iloc[0]

            bottom_height_diff = abs(processed_item.BottomHeightY - csv_values['BottomHeightY'])
            top_height_diff = abs(processed_item.TopHeightY - csv_values['TopHeightY'])
            if processed_item.ContaminationHeight < 0:
                contamination_height_diff = abs(csv_values['ContaminationHeight'] + processed_item.ContaminationHeight)
            else:
                contamination_height_diff = abs(processed_item.ContaminationHeight - csv_values['ContaminationHeight'])

            similarity_threshold = 25  # Threshold for similarity between processed and CSV values
            
            if (
                bottom_height_diff <= similarity_threshold
                and top_height_diff <= similarity_threshold and 
                contamination_height_diff <= similarity_threshold
            ):
                # calculate the ERROR 
                succesed += 1
                # FILL DICTIONARY ERROR MEASUREMENT ----------------------------------
                if processed_item.ContaminationHeight == 0:
                    continue
                errorDict['height'].append(contamination_height_diff)
                errorDict['top'].append(top_height_diff)
                errorDict['bottom'].append(bottom_height_diff)
                # END OF FILLING DICT ERROR MEASUREMENT ---------------------------
                if DEBUG:
                    print(Fore.GREEN + f"Values for {image_name} are close to the CSV values." + Style.RESET_ALL)
            else:
                if csv_values['ContaminationHeight'] == 0 or processed_item.ContaminationHeight == 0:
                    BaddlyValuatedNonContaminated['height'].append(processed_item.ContaminationHeight)
                    Zero_Contamination_Badly_Measured += 1
                    continue
                else:
                    failed += 1
                if DEBUG:
                    print (f"BottomHeightY: {processed_item.BottomHeightY} vs {csv_values['BottomHeightY']}")
                    print (f"TopHeightY: {processed_item.TopHeightY} vs {csv_values['TopHeightY']}")
                    print (f"ContaminationHeight: {processed_item.ContaminationHeight} vs {csv_values['ContaminationHeight']}")
                    print (f"BottomHeightY diff: {bottom_height_diff}")
                    print (f"TopHeightY diff: {top_height_diff}")
                    print (f"ContaminationHeight diff: {contamination_height_diff}")
                    print(Fore.RED + f"Values for {image_name} are not close to the CSV values. Image copied to failed images folder." + Style.RESET_ALL)

                if store_failed_images:
                    # Store the image in the folder for failed images
                    folder_with_images = os.path.join(current_directory, "Data_Storage", "others", "WholeDataset")
                    folder_for_failed_images = os.path.join(current_directory, "FailedImages")
                    image_path = os.path.join(folder_with_images, image_name)  # Assuming image is in this folder

                    # Create the directory if it doesn't exist
                    if not os.path.exists(folder_for_failed_images):
                        os.makedirs(folder_for_failed_images)

                    # Copy the image to the failed images folder
                    shutil.copy(image_path, folder_for_failed_images)
        else:
            print(f"No data found in CSV for {image_name}")

    print(f"Test results: {succesed} succesed, {failed} failed, {Zero_Contamination_Badly_Measured} zero contamination badly measured.")

    def plot_error_metrics(errorDict):
        # Extract the error metrics
        mae = {key: np.mean(values) for key, values in errorDict.items()}
        mse = {key: np.mean([v**2 for v in values]) for key, values in errorDict.items()}
        rmse = {key: np.sqrt(np.mean([v**2 for v in values])) for key, values in errorDict.items()}

        # Prepare data for plotting
        labels = list(mae.keys())
        mae_values = list(mae.values())
        mse_values = list(mse.values())
        rmse_values = list(rmse.values())
        
        x = np.arange(len(labels))  # the label locations
        width = 0.25  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width, mae_values, width, label='MAE')
        rects2 = ax.bar(x, mse_values, width, label='MSE')
        rects3 = ax.bar(x + width, rmse_values, width, label='RMSE')

        # Add some text for labels, title, and custom x-axis tick labels, etc.
        ax.set_xlabel('Measurement Type')
        ax.set_ylabel('Error Values')
        ax.set_title('Error Metrics Comparison by Measurement Type')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        # Attach a text label above each bar in *rects*, displaying its height.
        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(round(height, 2)),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)

        fig.tight_layout()
        plt.savefig('error_metrics_comparison.png')  # Save the figure
        plt.show()


    # count the number of Badly Valuated Non Contaminated images that are not zero
    count = 0
    for value in BaddlyValuatedNonContaminated['height']:
        if value != 0:
            count += 1
            
    print ("Number of Badly Valuated Non Contaminated images that are not zero: ", count,"out of ", len(BaddlyValuatedNonContaminated['height']))

    # plot_error_metrics(errorDict)

    # CALCULATE THE ERROR MEASUREMENT ----------------------------------
    try :
        mae = {key: np.mean(values) for key, values in errorDict.items()}

        # If you also need Mean Squared Error (MSE)
        mse = {key: np.mean([v**2 for v in values]) for key, values in errorDict.items()}

        # And Root Mean Squared Error (RMSE)
        rmse = {key: np.sqrt(np.mean([v**2 for v in values])) for key, values in errorDict.items()}
        
        median = {key: np.median(values) for key, values in errorDict.items()}
    except:
        print ("Error in calculating error measurement.")
    # END OF CALCULATING ERROR MEASUREMENT ---------------------------

    return mae, mse, rmse, median, succesed, failed, Zero_Contamination_Badly_Measured

