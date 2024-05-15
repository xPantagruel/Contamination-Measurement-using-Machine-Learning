# @file ContaminationMeasurementClass.py
# @brief main functionality behind the measurement of the contamination height
# @author Matěj Macek (xmacek27@stud.fit.vutbr.cz)
# @date 4.5.2024

from matplotlib.pyplot import imshow
from image_processing import *

class ContaminationMeasurementClass:
    def measure_contamination(self, image_path, DEBUG=False, showCannyVsScharr=False,plot=False,Store_Images_with_detected_lines=False,plot_all = False):
        image = load_image(image_path)
        OpenImage = apply_opening(image)
        CloseImage = apply_closing(OpenImage)

        kernel_size = 11
        blurred_image = cv2.GaussianBlur(
            CloseImage, (kernel_size, kernel_size), 0)

        thresholded_image = otsu_thresholding(blurred_image)
        scharr_image = self.scharr(thresholded_image)
        
        if showCannyVsScharr:
            canny_image = cv2.Canny(thresholded_image, 45, 50)

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            # Display the original image
            axs[0].imshow(image, cmap='gray')
            axs[0].set_title('Original Image', fontsize=22)
            axs[0].axis('off')

            # Display the image after applying Canny edge detection
            axs[1].imshow(canny_image, cmap='gray')
            axs[1].set_title('Canny Edge Detection', fontsize=22)
            axs[1].axis('off')

            # Display the image after applying Scharr edge detection
            axs[2].imshow(scharr_image, cmap='gray')
            axs[2].set_title('Scharr Edge Detection', fontsize=22)
            axs[2].axis('off')

            # Adjust the layout
            plt.tight_layout()

            plt.show()
    
        TinBallEdgeLeft = get_mode_height_of_tin_ball_left_side(scharr_image)
        TinBallEdgeRight = get_mode_height_of_tin_ball_right_side(scharr_image)
        
        # Get high of tin ball
        if (TinBallEdgeLeft > TinBallEdgeRight):
            MaxY = TinBallEdgeLeft
        else:
            MaxY = TinBallEdgeRight
        if DEBUG:
            print("MaxY: ", MaxY)
        
        LeftYContamination, RightYContamination, middleOfContamination = get_contamination_range(
            scharr_image, MaxY)
        
        if DEBUG:
            print("LeftYContamination: ", LeftYContamination)
            print("RightYContamination: ", RightYContamination)
        
        # zero contamination check 
        if LeftYContamination == -1 or RightYContamination == -1:
            return 0,0

        if RightYContamination - LeftYContamination < 200:
            # i need to edit these values sot between them is 400 pixels
            middle_between_edges = ((RightYContamination + LeftYContamination) // 2) + LeftYContamination
            LeftYContamination = middle_between_edges - 200
            RightYContamination = middle_between_edges + 200
        
        roi = get_Roi(blurred_image, LeftYContamination, RightYContamination)
        starting_point = get_starting_point_TEST(roi, showDebug=plot, TinBallEdge = MaxY)

        if DEBUG:
            print("starting_point: ", starting_point)

        maxs, mins, bottom_of_contamination, top_of_contamination = find_contamination_bottom_and_top(roi, starting_point,shouwDebug=plot)
        
        if Store_Images_with_detected_lines:
            if not os.path.exists('test'):
                os.makedirs('test')
            image_to_save = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            cv2.line(image_to_save, (0, top_of_contamination), (image_to_save.shape[1], top_of_contamination), (0, 255, 0), 2)
            cv2.line(image_to_save, (0, bottom_of_contamination), (image_to_save.shape[1], bottom_of_contamination), (0, 0, 255), 2)

            cv2.imwrite('test/' + os.path.basename(image_path) + 'test' + '.png', image_to_save )

        # Visualization
        if plot_all:
            images_to_visualize = [image,
                                CloseImage, thresholded_image, scharr_image,roi]
            titles = ["Original Image",
                    "CloseImage", "Thresholded Image", "Scharr Edge Detection","ROI"]

            self.visualize(images_to_visualize, titles)
        return top_of_contamination, bottom_of_contamination

    def showHistogram(self, image_path):
        plot_histogram(image_path)


    def blur_image(self, image):
        return blurring(image)

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