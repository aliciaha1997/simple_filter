# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:42:06 2021

@author: Alicia Ngoc Diep Ha
"""

import cv2
import numpy as np

# OnChange event handler
def dummy(value):
    pass

# Define convolution kernels
identity_kernel = np.array([[0,0,0],[0,1,0],[0,0,0]])
sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
gaussian_kernel1 = cv2.getGaussianKernel(3,0)
gaussian_kernel2 = cv2.getGaussianKernel(5,0) # more blur
box_kernel = np.array([[1,1,1],[1,1,1],[1,1,1]], np.float32) / 9.0
 

kernels = [identity_kernel, sharpen_kernel, gaussian_kernel1, gaussian_kernel2, box_kernel]

# Read image, grayscale copy
colour_original = cv2.imread('test.jpg')
colour_grayscale = cv2.cvtColor(colour_original, cv2.COLOR_BGR2GRAY)

# UI (window and trackbars)
cv2.namedWindow('Image Editor App') # Create window w/ name

# Track Bars
cv2.createTrackbar('Contrast', 'Image Editor App', 1, 5, dummy)
cv2.createTrackbar('Brightness', 'Image Editor App', 50, 100, dummy) # Add/remove brightness
cv2.createTrackbar('Filter', 'Image Editor App', 0, len(kernels)-1, dummy) # TODO: update max value to number of filters
cv2.createTrackbar('Grayscale', 'Image Editor App', 0, 1, dummy)

# Counter for image number
count = 1

# Main UI Loop
while True:
    
    grayscale = cv2.getTrackbarPos('Grayscale', 'Image Editor App')
    contrast = cv2.getTrackbarPos('Contrast', 'Image Editor App')
    brightness = cv2.getTrackbarPos('Brightness', 'Image Editor App')
    kernel_idx = cv2.getTrackbarPos('Filter', 'Image Editor App')
    
    # Apply the filters
    colour_modified = cv2.filter2D(colour_original, -1, kernels[kernel_idx])
    gray_modified = cv2.filter2D(colour_grayscale, -1, kernels[kernel_idx])
    
    # Apply the brightness/contrast
    colour_modified = cv2.addWeighted(colour_modified, contrast, 
                                      np.zeros_like(colour_original), 
                                      0, brightness - 50)
    
    gray_modified = cv2.addWeighted(gray_modified, contrast, 
                                      np.zeros_like(colour_grayscale), 
                                      0, brightness - 50)
    
    # Keypress
    key = cv2.waitKey(100)
    
    # Quit
    if key == ord('q'): # Convert char into int
        break
        
    # Save
    elif key == ord('s'):
        if grayscale == 0:
            cv2.imwrite("output-{}.png".format(count), colour_modified)
        else:
            cv2.imwrite("output-{}.png".format(count), gray_modified)
        count += 1
    
    # Show the image
    if grayscale == 0:
        cv2.imshow('Image Editor App', colour_modified)
    else:
        cv2.imshow('Image Editor App', gray_modified)

cv2.destroyAllWindows() # Window cleanup