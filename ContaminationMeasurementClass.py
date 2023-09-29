from matplotlib.pyplot import imshow
from image_processing import *
from helper_class import *

class ContaminationMeasurementClass:
    def measure_contamination(self, image_path):
        image = load_image(image_path)
        preprocessed_image = self.cutt_off_edges(image)
        blurred_image = self.blur_image(preprocessed_image)
        thresholded_image1 = thresholding(blurred_image, 100, 255, cv2.THRESH_BINARY_INV)
        adaptive_threshold_image = adaptive_threshold(thresholded_image1)  
        thresholded_image = thresholding(adaptive_threshold_image)
        # thresholded_image = self.threshold_image(thresholded_image)
        
        canny_image = self.detect_edges(thresholded_image)
        scharr_image = self.scharr(thresholded_image)

        # Visualization
        images_to_visualize = [image, preprocessed_image, blurred_image, thresholded_image, scharr_image,]
        titles = ["Original Image", "Preprocessed Image", "Blurred Image", "Thresholded Image", "Scharr Edge Detection"]

        self.visualize(images_to_visualize, titles)

        # image = load_image(image_path)
        # preprocessed_image = self.cutt_off_edges(image)
        # adaptive_threshold_gui(preprocessed_image)
    
    def cutt_off_edges(self, image):
        return preprocess_image(image)
    
    def blur_image(self, image):
        return blurring(image) 
    
    def detect_edges(self, image):
        return canny_edge_detection(image, 45, 50, 7, 2)
    
    def probabilistic_hough_lines_example(self,image):
        # Define Hough parameters
        rho = 1  # Pixel resolution of the accumulator
        theta = np.pi / 180  # Angle resolution of the accumulator (1 degree)
        threshold = 50  # Accumulator threshold (adjust as needed)
        min_line_length = 100  # Minimum line length to be considered
        max_line_gap = 10  # Maximum gap between line segments to be considered as a single line

        # Detect lines using Probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(image, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

        # Draw detected lines on the image (for visualization)
        line_image = np.zeros_like(image)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)  # White color

        return line_image
        # # Display the image with detected lines
        # cv2.imshow("Probabilistic Hough Lines", line_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    def detect_horizontal_lines(self, canny_image, original_image):
        # Copy the original image to avoid modifying it
        highlighted_image = original_image.copy()

        # Use Hough Line Transform to detect lines
        lines = cv2.HoughLines(canny_image, 1, np.pi / 180, threshold=100)

        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                if np.pi / 4 < theta < 3 * np.pi / 4:  # Check if the line is roughly horizontal
                    # Calculate the line's endpoints
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho

                    # Draw the line on the image
                    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                    cv2.line(highlighted_image, pt1, pt2, (0, 0, 255), 2)  # Red color for lines

        return highlighted_image
    
    def scharr(self,image):
        # Apply Scharr filter along x-axis
        gradient_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)

        # Apply Scharr filter along y-axis
        gradient_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)

        # Calculate the gradient magnitude
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

        # Normalize the gradient magnitude
        gradient_magnitude *= 255.0 / gradient_magnitude.max()

        # Convert back to uint8
        gradient_magnitude = np.uint8(gradient_magnitude)

        return gradient_magnitude
    
    def visualize(self, image_list, title_list):
        # Display the images
        plt.figure(figsize=(12, 6))

        for i in range(len(image_list)):
            plt.subplot(2, (len(image_list) + 1) // 2, i + 1)
            if len(image_list[i].shape) == 2:  # Grayscale image
                plt.imshow(image_list[i], cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(image_list[i], cv2.COLOR_BGR2RGB))
            plt.title(title_list[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()