import cv2
import pandas as pd
import tkinter as tk
import os
import shutil
import image_processing as image_processing

class ImageThresholding:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        self.threshold_value = 128
        self.max_value = 255
        self.threshold_type = cv2.THRESH_BINARY
        self.threshold_type_name = "Threshold Type: BINARY"
        self.window_name = 'Thresholding'
        self.image_name = os.path.basename(self.image_path)  # Extracting only image name from full path
        self.csv_filename = 'threshold_values.csv'

    def apply_threshold(self):
        _, thresholded = cv2.threshold(self.image, self.threshold_value, self.max_value, self.threshold_type)
        cv2.imshow(self.window_name, thresholded)

    def on_trackbar(self, val):
        self.threshold_value = val
        self.apply_threshold()

    def on_type_change(self, val):
        if val == 0:
            self.threshold_type = cv2.THRESH_BINARY
            self.threshold_type_name = "Threshold Type: BINARY"
        elif val == 1:
            self.threshold_type = cv2.THRESH_BINARY_INV
            self.threshold_type_name = "Threshold Type: BINARY_INV"
        elif val == 2:
            self.threshold_type = cv2.THRESH_TRUNC
            self.threshold_type_name = "Threshold Type: TRUNC"
        elif val == 3:
            self.threshold_type = cv2.THRESH_TOZERO
            self.threshold_type_name = "Threshold Type: TOZERO"
        elif val == 4:
            self.threshold_type = cv2.THRESH_TOZERO_INV
            self.threshold_type_name = "Threshold Type: TOZERO_INV"

        self.apply_threshold()
        cv2.displayOverlay(self.window_name, self.threshold_type_name, 1000)

    def create_trackbars(self):
        cv2.createTrackbar('Threshold Value', self.window_name, self.threshold_value, self.max_value, self.on_trackbar)
        cv2.createTrackbar('Threshold Type', self.window_name, 0, 4, self.on_type_change)

    def save_values_to_csv(self):
        if os.path.exists(self.csv_filename):
            df = pd.read_csv(self.csv_filename)

            if self.image_name in df['Image Name'].values:
                df.loc[df['Image Name'] == self.image_name, self.threshold_type_name] = self.threshold_value
            else:
                new_row = pd.DataFrame({
                    'Image Name': [self.image_name],
                    'Threshold Type: BINARY': [0],
                    'Threshold Type: BINARY_INV': [0],
                    'Threshold Type: TRUNC': [0],
                    'Threshold Type: TOZERO': [0],
                    'Threshold Type: TOZERO_INV': [0]
                })
                new_row[self.threshold_type_name] = self.threshold_value
                df = pd.concat([df, new_row], ignore_index=True)

            df.to_csv(self.csv_filename, index=False)
        else:
            data = {
                'Image Name': [self.image_name],
                'Threshold Type: BINARY': [0],
                'Threshold Type: BINARY_INV': [0],
                'Threshold Type: TRUNC': [0],
                'Threshold Type: TOZERO': [0],
                'Threshold Type: TOZERO_INV': [0]
            }
            df = pd.DataFrame(data)
            df[self.threshold_type_name] = self.threshold_value
            df.to_csv(self.csv_filename, index=False)

        print("Threshold values saved to threshold_values.csv")

    def save_image(self):
        output_folder = 'output_images'  # Define the output folder where images will be saved
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_image_path = os.path.join(output_folder, self.image_name)
        cv2.imwrite(output_image_path, self.image)
        print(f"Image saved to: {output_image_path}")

    def run(self):
        self.image = image_processing.preprocess_image(self.image)
        cv2.namedWindow(self.window_name)
        self.apply_threshold()
        self.create_trackbars()

        while True:
            key = cv2.waitKey(1)
            if key == 27:  # Esc key to exit
                break
            elif key == ord('i'):  # Press 'i' to save the image
                self.save_image()
            elif key == ord('s'):  # Press 's' to save current values to CSV
                self.save_values_to_csv()

        cv2.destroyAllWindows()

# Tkinter GUI for Image Thresholding
def threshold_image(image_path):
    thresholding = ImageThresholding(image_path)

    def save_to_csv():
        thresholding.save_values_to_csv()

    root = tk.Tk()
    root.title("Image Thresholding")
    root.geometry("400x100")

    button_csv = tk.Button(root, text="Save Values to CSV", command=save_to_csv)
    button_csv.pack()

    thresholding.run()

