import os
import pandas as pd
from colorama import Fore, Style  # Import colorama modules for text coloring


def test_csv_data(processed_data):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    folder_path = os.path.join(current_directory, "images")
    csv_file = os.path.join(folder_path, "ImagesContaminationResults.csv")
    data = pd.read_csv(csv_file)

    succesed = 0
    failed = 0
    # Loop through processed data and compare with CSV
    for processed_item in processed_data:
        image_name = processed_item.ImageName
        csv_row = data[data['ImageName'] == image_name]

        if not csv_row.empty:
            csv_values = csv_row.iloc[0]

            bottom_height_diff = abs(processed_item.BottomHeightY - csv_values['BottomHeightY'])
            top_height_diff = abs(processed_item.TopHeightY - csv_values['TopHeightY'])
            contamination_height_diff = abs(processed_item.ContaminationHeight - csv_values['ContaminationHeight'])

            similarity_threshold = 15



            if (
                bottom_height_diff <= similarity_threshold
                and top_height_diff <= similarity_threshold
                and contamination_height_diff <= similarity_threshold
            ):
                succesed += 1
                print(Fore.GREEN + f"Values for {image_name} are close to the CSV values." + Style.RESET_ALL)
            else:
                failed += 1
                print(Fore.RED + f"Values for {image_name} are not close to the CSV values." + Style.RESET_ALL)
        else:
            print(f"No data found in CSV for {image_name}")

    print(f"Test results: {succesed} succesed, {failed} failed")
