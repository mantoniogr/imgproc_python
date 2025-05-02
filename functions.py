#!/usr/bin/env python3

"""
Image processing utility functions for basic operations.
Includes grayscale conversion, negative image creation,
thresholding, and object counting.
"""

#
#  functions.py
#  imgproc_python
#
#  Created by Marco Garduno on 24/07/17.
#  Copyright 2017 Marco Garduno. All rights reserved.
#

import cv2
import numpy as np

# RGB weights for grayscale conversion
RGB_WEIGHTS = np.array([0.114, 0.587, 0.299])
MAX_PIXEL_VALUE = 255

def rgb2gray(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale.
    
    Args:
        image (np.ndarray): Input RGB image array of shape (height, width, 3)
        
    Returns:
        np.ndarray: Grayscale image array of shape (height, width) with uint8 dtype
    """
    if image.ndim != 3:
        raise ValueError("Input image must be a 3-channel RGB image")
    if image.shape[2] != 3:
        raise ValueError("Input image must have 3 channels")
    
    # Convert to grayscale and ensure uint8 data type
    return np.dot(image[..., :3], RGB_WEIGHTS).astype(np.uint8)

def negative_gray(image: np.ndarray) -> np.ndarray:
    """Create negative of a grayscale image."""
    return MAX_PIXEL_VALUE - image

def negative_color(image: np.ndarray) -> np.ndarray:
    """Create negative of a color image."""
    return MAX_PIXEL_VALUE - image

def threshold(image: np.ndarray, th1: int) -> np.ndarray:
    """Apply a binary threshold to a grayscale image.
    
    Args:
        image (np.ndarray): Input grayscale image array of shape (height, width)
        th1 (int): Threshold value (0-255)
        
    Returns:
        np.ndarray: Binary image array of shape (height, width) with uint8 dtype
    """
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale image")
    
    if not (0 <= th1 <= 255):
        raise ValueError("Threshold value must be in the range [0, 255]")
    
    # Create a binary image based on the threshold
    binary_image = np.where(image > th1, 255, 0).astype(np.uint8)
    
    return binary_image

def threshold_range(image: np.ndarray, th1: int, th2: int) -> np.ndarray:
    """Apply a binary threshold to a grayscale image within a range.
    
    Args:
        image (np.ndarray): Input grayscale image array of shape (height, width)
        th1 (int): Lower threshold value (0-255)
        th2 (int): Upper threshold value (0-255)    
        
    Returns:
        np.ndarray: Binary image array of shape (height, width) with uint8 dtype
    """
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale image")
    
    height, width =  image.shape
    img = np.copy(image)

    for j in range(0, height):
        for i in  range(0, width):
            if image[j,i] > th1 and th2 > image[j,i]:
                img[j,i] = 255;
            else:
                img[j,i] = 0;

    return img

def counting_objects(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Count objects in a binary image.
    
    Args:
        image (np.ndarray): Input binary image array of shape (height, width)
        
    Returns:
        tuple[np.ndarray, int]: Tuple containing the labeled image and the number of objects
    """
    aux = np.copy(image)
    height, width =  image.shape

    k = 0
    fifo = []

    for j in range(0, height):
        for i in range(0, width):
            if image[j,i] != 0:
                k = k + 10
                fifo.append([j,i])
                image[j,i] = 0
                aux[j,i] = k
                while(fifo):
                    primas = fifo.pop(0)
                    for n in range(primas[0] - 1, primas[0] + 2):
                        for m in range(primas[1] - 1, primas[1] + 2):
                            if image[n,m] != 0:
                                fifo.append([n,m])
                                image[n,m] = 0
                                aux[n,m] = k

    return aux, k
