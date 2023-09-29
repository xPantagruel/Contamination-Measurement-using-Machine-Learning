import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

def adaptive_threshold_gui(image):
    def update_threshold_image():
        block_size = int(block_size_slider.get())  # Cast to integer
        c = int(c_slider.get())  # Cast to integer

        thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
        print(block_size, c)
        thresholded_image = Image.fromarray(thresholded)
        thresholded_photo = ImageTk.PhotoImage(image=thresholded_image)

        thresholded_label.config(image=thresholded_photo)
        thresholded_label.image = thresholded_photo


    # Create the main window
    root = tk.Tk()
    root.title('Adaptive Thresholding')

    # Create a frame for sliders
    slider_frame = ttk.Frame(root)
    slider_frame.pack(padx=10, pady=10)

    # Create sliders for block_size and c
    block_size_slider = ttk.Scale(slider_frame, from_=3, to=31, length=200, orient='horizontal')
    c_slider = ttk.Scale(slider_frame, from_=-10, to=10, length=200, orient='horizontal')

    # Set initial values for sliders
    block_size_slider.set(31)
    c_slider.set(-10)

    # Create a button to apply thresholding
    apply_button = ttk.Button(root, text='Apply Thresholding', command=update_threshold_image)

    # Create a label to display the thresholded image
    thresholded_label = ttk.Label(root)

    # Pack the sliders, button, and label
    block_size_slider.pack()
    c_slider.pack()
    apply_button.pack()
    thresholded_label.pack()

    # Update the thresholded image initially
    update_threshold_image()

    root.mainloop()
