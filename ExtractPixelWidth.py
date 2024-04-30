
import csv
import re
import os

def extract_metadata_value(metadata_block, key):
    # Use regex to find the key and extract its value
    match = re.search(r'^' + re.escape(key) + r'=(.*)$', metadata_block, re.M)
    if match:
        return match.group(1).strip()
    return None

def process_tiff_metadata(file_path, csv_file):
    # Read the entire file content
    with open(file_path, 'rb') as file:
        content = file.read().decode('iso-8859-1')  # Adjust encoding as necessary

    # Find the start and end of the metadata block
    start = content.find('Date=')
    end = content.rfind('Rotation=0') + len('Rotation=0')
    
    if start == -1 or end == -1:
        print(f"Metadata block not found in {file_path}.")
        return None
    
    # Extract the metadata block
    metadata_block = content[start:end]

    # Extract PixelWidth from metadata
    pixel_width = extract_metadata_value(metadata_block, 'PixelWidth')
    
    # Store results in CSV, removing the file extension from the filename
    filename_without_extension = os.path.splitext(os.path.basename(file_path))[0]
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([filename_without_extension, pixel_width])

def process_directory(directory_path, csv_file):
    # Ensure the CSV file has the correct headers before writing data
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'PixelWidth'])
    
    # Process each TIFF file in the directory
    for file in os.listdir(directory_path):
        if file.lower().endswith('.tif'):
            full_path = os.path.join(directory_path, file)
            process_tiff_metadata(full_path, csv_file)

# Path to the directory containing the TIFF files
# directory_path = r'C:\Users\matej.macek\OneDrive - Thermo Fisher Scientific\Desktop\TiffImages'
directory_path = r'C:\Users\matej.macek\OneDrive - Thermo Fisher Scientific\Desktop\NewImages\UniqueImages2'
# Path to the CSV file
csv_file = 'pixelWidth_Uniq.csv'

process_directory(directory_path, csv_file)

