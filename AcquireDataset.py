import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

class AcquireDataset:
    def __init__(self, path):
        self.path = path
        self.crop_mapping = {
            (1536, 1094): 185,
            (768, 547): 90,
            (3072, 2188): 370,
            (1024, 954): 145,
            (1024, 943): 57,
            (1024, 929): 110,
            (1024, 931): 116,
            (2048, 1908): 298,
        }
        self.lower_green = np.array([40, 40, 40])
        self.upper_green = np.array([80, 255, 255])

        # Define ranges for red color
        self.lower_red1 = np.array([0, 50, 50])
        self.upper_red1 = np.array([10, 255, 255])
        
        self.lower_red2 = np.array([160, 50, 50])
        self.upper_red2 = np.array([180, 255, 255])

        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([40, 255, 255])


    def load_images_from_folder(self):
        # List all files in the directory
        all_files = os.listdir(self.path)
        image_files = [f for f in all_files if f.endswith(('.tif'))]  # Add other formats if needed

        # Lists to store the data
        image_names = []
        image_sizes = []

        # Loop through all the image files
        for image_file in image_files:
            image_path = os.path.join(self.path, image_file)
            image = cv2.imread(image_path)
            if image is not None:  # Check if the image is read successfully
                h, w, _ = image.shape
                image_names.append(image_file)
                image_sizes.append((w, h))

        # Create a dataframe
        data = {
            'image_name': image_names,
            'image_size': image_sizes
        }
        df_size = pd.DataFrame(data)

        # Display the dataframe
        print(df_size)

        unique_sizes =df_size['image_size'].value_counts()

        print("Unique Image Sizes and their Counts:")
        print(unique_sizes)

    def crop_images_in_directory(self):
        # List all image files in the directory
        all_files = os.listdir(self.path)
        image_files = [f for f in all_files if f.endswith(('.tif', '.jpg', '.png'))]  # Add other formats if needed
        
        for image_file in image_files:
            image_path = os.path.join(self.path, image_file)
            
            # Read the image
            image = cv2.imread(image_path)
            
            # Check if the image was loaded correctly
            if image is None:
                print(f"Error: Couldn't load {image_file}. Skipping...")
                continue
            
            # Get the size of the image
            h, w = image.shape[:2]
            
            # Determine the amount to crop based on the size
            pixel_amount = self.crop_mapping.get((w, h), None)
            
            if pixel_amount:
                # Calculate the new height
                new_height = h - pixel_amount
                
                # Crop the image
                cropped_image = image[:new_height, :]
                
                # Save the cropped image (in-place)
                cv2.imwrite(image_path, cropped_image)
                print(f"Cropped {image_file}")
            else:
                print(f"Size {w}x{h} for {image_file} not found in mapping. Skipping...")


    def create_combined_mask(self, image_path):
        def create_mask(image_path, lower_color, upper_color):
            # Read the image
            image = cv2.imread(image_path)

            # Convert BGR to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Create masks for each range of red color
            mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
            mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)

            # Combine red masks into one
            mask_red = cv2.bitwise_or(mask1, mask2)

            # Create masks for green and yellow colors
            mask_green = cv2.inRange(hsv, self.lower_green, self.upper_green)
            mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

            # Combine masks into one by bitwise OR operation
            combined_mask = cv2.bitwise_or(mask_green, cv2.bitwise_or(mask_red, mask_yellow))

            # Apply dilation to the combined mask
            kernel = np.ones((3,3), np.uint8)
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
            cv2.imshow('combined_mask', combined_mask)
            cv2.waitKey(0)
            cv2.imwrite(r'C:\Users\matej.macek\OneDrive - Thermo Fisher Scientific\Desktop\BC Contamination Measurement\BC- FORK\ContaminationMeasurement\combined_mask.png', combined_mask)
            return combined_mask

        # Create combined mask using the provided image path
        result_combined_mask = create_mask(image_path, self.lower_red1, self.upper_red1)

        return result_combined_mask

    def Get_Image_Without_Dimensions(self, image_path):
        # Create the combined mask using the provided image path
        result_combined_mask = self.create_combined_mask(image_path)

        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        result_image = cv2.inpaint(original_image, result_combined_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
        cv2.imshow('result_image', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(r'C:\Users\matej.macek\OneDrive - Thermo Fisher Scientific\Desktop\BC Contamination Measurement\BC- FORK\ContaminationMeasurement\result_image.png', result_image)
        return result_image
    
    def Remove_Dimensions_From_Images(self):
        # List all image files in the directory
        all_files = os.listdir(self.path)
        image_files = [f for f in all_files if f.endswith(('.tif', '.jpg', '.png'))]
        #create direcotry for images without dimensions
        new_path = self.path + r'\ImagesWithoutDimensions'
        os.mkdir(new_path)

        for image_file in image_files:
            image_path = os.path.join(self.path, image_file)
            # remove dimensions from image
            result_image = self.Get_Image_Without_Dimensions(image_path)
            #save image
            cv2.imwrite(os.path.join(new_path, image_file), result_image)


# image_path = r'C:\Users\matej.macek\OneDrive - Thermo Fisher Scientific\Desktop\BC Contamination Measurement\BC- FORK\ContaminationMeasurement\with quotation\H6EX10_S_PLC_SPC_Upload_01.tif'
image_path = r'C:\Users\matej.macek\OneDrive - Thermo Fisher Scientific\Desktop\BC Contamination Measurement\BC- FORK\ContaminationMeasurement\ImagesWithDimensions\H6EX10_S_PLC_SPC_Upload_01.tif'

AcquireDataset = AcquireDataset(r'C:\Users\matej.macek\OneDrive - Thermo Fisher Scientific\Desktop\BC Contamination Measurement\BC- FORK\ContaminationMeasurement\ImagesWithDimensions')

AcquireDataset.Get_Image_Without_Dimensions(image_path)
# AcquireDataset.Remove_Dimensions_From_Images()
