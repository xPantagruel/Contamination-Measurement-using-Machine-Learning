from matplotlib.pyplot import imshow
from image_processing import *
from helper_class import *

class ContaminationMeasurementClass:
    def measure_contamination(self, image_path):
        image = load_image(image_path)
        preprocessed_image = self.cutt_off_edges(image)
        kernel_size = 11
        blurred_image = cv2.GaussianBlur(
            preprocessed_image, (kernel_size, kernel_size), 0)
        thresholded_image1 = thresholding(
            blurred_image, 96, 255, cv2.THRESH_TOZERO)
        thresholded_image = thresholding(
            thresholded_image1, 100, 255, cv2.THRESH_TRUNC)

        canny_image = self.detect_edges(thresholded_image)
        scharr_image = self.scharr(thresholded_image)

        # Visualization
        images_to_visualize = [image,
                               thresholded_image1, thresholded_image, scharr_image, thresholded_image1]
        titles = ["Original Image",
                  "thresholded_image1", "Thresholded Image", "Scharr Edge Detection", "thresholded_image1"]

        self.visualize(images_to_visualize, titles)

    def measure_contamination2(self, image_path):
        image = load_image(image_path)
        preprocessed_image = self.cutt_off_edges(image)
        # removed_background_image, edge_values = remove_background_above_Tin_Ball(
        #     preprocessed_image, threshold=260)

        # start_indices, end_indices = find_start_end_indices(edge_values)
        # print("Start Indices:", start_indices)
        # print("End Indices:", end_indices)
        kernel_size = 11
        blurred_image = cv2.GaussianBlur(
            preprocessed_image, (kernel_size, kernel_size), 0)
        thresholded_image1 = thresholding(
            blurred_image, 96, 255, cv2.THRESH_TOZERO)
        thresholded_image = thresholding(
            thresholded_image1, 100, 255, cv2.THRESH_TRUNC)
        removed_background_image, edge_values = remove_background_above_Tin_Ball(
            thresholded_image, threshold=260)
        canny_image = self.detect_edges(thresholded_image)
        scharr_image = self.scharr(removed_background_image)

        # Visualization
        images_to_visualize = [image,
                               thresholded_image1, thresholded_image, scharr_image, removed_background_image]
        titles = ["Original Image",
                  "thresholded_image1", "Thresholded Image", "Scharr Edge Detection", "removed_background_image"]

        self.visualize(images_to_visualize, titles)

    def measure_contamination3(self, image_path):
        image = load_image(image_path)
        preprocessed_image = self.cutt_off_edges(image)
        # removed_background_image, edge_values = remove_background_above_Tin_Ball(
        #     preprocessed_image, threshold=260)
        kernel_size = 11
        blurred_image = cv2.GaussianBlur(
            preprocessed_image, (kernel_size, kernel_size), 0)
        thresholded_image1 = thresholding(
            blurred_image, 96, 255, cv2.THRESH_TOZERO)
        thresholded_image = thresholding(
            thresholded_image1, 100, 255, cv2.THRESH_TRUNC)
        # Replace "images" with "removed_background_image" in the image path
        new_image_path = os.path.join(os.path.dirname(
            image_path), "thresholdedImages", os.path.basename(image_path))

        # Save the processed image
        # Assuming 'removed_background_image' is a valid NumPy array
        save_image(thresholded_image, new_image_path)

    def measure_contamination4(self, image_path):
        image = load_image(image_path)
        preprocessed_image = self.cutt_off_edges(image)
        canny_image = canny_edge_detection(
            preprocessed_image.copy(), 45, 50, 7, 2)
        # contours = find_and_draw_contours(canny_image, preprocessed_image)
        # Get minimum of tin ball in image height
        TinBallEdgeLeft = get_mode_height_of_tin_ball_left_side(canny_image)
        TinBallEdgeRight = get_mode_height_of_tin_ball_right_side(canny_image)

        print("TinBallEdgeLeft: ", TinBallEdgeLeft)
        print("TinBallEdgeRight: ", TinBallEdgeRight)

        kernel_size = 11
        blurred_image = cv2.GaussianBlur(
            preprocessed_image, (kernel_size, kernel_size), 0)
        thresholded_image1 = thresholding(
            blurred_image, 96, 255, cv2.THRESH_TOZERO)
        thresholded_image = thresholding(
            thresholded_image1, 100, 255, cv2.THRESH_TRUNC)

        canny_image2 = self.detect_edges(thresholded_image)
        scharr_image2 = self.scharr(thresholded_image)

        # Get high of tin ball
        if (TinBallEdgeLeft > TinBallEdgeRight):
            MaxY = TinBallEdgeLeft
        else:
            MaxY = TinBallEdgeRight

        print("MaxY: ", MaxY)
        BottomOfContamination = get_contamination_bottom_height(
            scharr_image2, MaxY)
        print("contamination_bottom: ", BottomOfContamination)

        LeftSideContamination, RightSideContamination, middleOfContamination = get_contamination_range(
            scharr_image2, MaxY)
        
        TopContamination = Get_Contamination_Height(scharr_image2, middleOfContamination, BottomOfContamination, MaxY)
        
        ContaminationHeight = BottomOfContamination - TopContamination

        # # Visualization
        # images_to_visualize = [image,
        #                        blurred_image, thresholded_image, scharr_image2, canny_image2]
        # titles = ["Original Image",
        #           "blurred_image", "Thresholded Image", "Scharr Edge Detection", "canny_image"]

        # self.visualize(images_to_visualize, titles)

        # # Visualization of image with detected lines
        # VisualizeBottomAndTopOfContamination = image.copy()
        # cv2.line(VisualizeBottomAndTopOfContamination, (0, TopContamination),
        #          (VisualizeBottomAndTopOfContamination.shape[1], TopContamination), (255, 255, 255), 2)
        # cv2.line(VisualizeBottomAndTopOfContamination, (0, BottomOfContamination),
        #             (VisualizeBottomAndTopOfContamination.shape[1], BottomOfContamination), (255, 255, 255), 2)
        # cv2.imshow("VisualizeBottomAndTopOfContamination", VisualizeBottomAndTopOfContamination)
        # cv2.waitKey(0)


        return BottomOfContamination, TopContamination

    def cutt_off_edges(self, image):
        return preprocess_image(image)

    def blur_image(self, image):
        return blurring(image)

    def detect_edges(self, image):
        return canny_edge_detection(image, 45, 50, 7, 2)

    def probabilistic_hough_lines_example(self, image):
        # Define Hough parameters
        rho = 1  # Pixel resolution of the accumulator
        theta = np.pi / 180  # Angle resolution of the accumulator (1 degree)
        threshold = 50  # Accumulator threshold (adjust as needed)
        min_line_length = 100  # Minimum line length to be considered
        max_line_gap = 10  # Maximum gap between line segments to be considered as a single line

        # Detect lines using Probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(image, rho, theta, threshold,
                                minLineLength=min_line_length, maxLineGap=max_line_gap)

        # Draw detected lines on the image (for visualization)
        line_image = np.zeros_like(image)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2),
                     (255, 255, 255), 2)  # White color

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
                    cv2.line(highlighted_image, pt1, pt2,
                             (0, 0, 255), 2)  # Red color for lines

        return highlighted_image

    def scharr(self, image):
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