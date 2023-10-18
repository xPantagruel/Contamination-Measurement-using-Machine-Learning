import cv2
import numpy as np


class ImageThresholdingApp:
    def __init__(self, image):
        self.image = image
        if self.image is None:
            raise ValueError("Image not found or could not be loaded.")

        self.threshold_type = cv2.THRESH_TOZERO_INV
        self.threshold_value = 153

        self.create_window()

    def create_window(self):
        cv2.namedWindow("Image Thresholding", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Image Thresholding",
                              cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.createTrackbar("Threshold Value", "Image Thresholding",
                           self.threshold_value, 255, self.update_threshold)
        cv2.createTrackbar("Threshold Type", "Image Thresholding",
                           0, 4, self.update_threshold_type)
        self.update_threshold(0)  # Initial thresholding

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Press 'Esc' to exit
                break

        cv2.destroyAllWindows()

    def update_threshold(self, value):
        self.threshold_value = cv2.getTrackbarPos(
            "Threshold Value", "Image Thresholding")
        _, thresholded_image = cv2.threshold(
            self.image, self.threshold_value, 255, self.threshold_type)
        cv2.imshow("Image Thresholding", thresholded_image)

    def update_threshold_type(self, value):
        threshold_types = [cv2.THRESH_TOZERO_INV, cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV,
                           cv2.THRESH_TRUNC, cv2.THRESH_TOZERO]
        self.threshold_type = threshold_types[value]
        # Update the thresholded image when the type changes
        self.update_threshold(0)


if __name__ == "__main__":
    image_path = "your_image.jpg"  # Replace with your image path
    app = ImageThresholdingApp(image_path)
