#!/usr/bin/env python3

#
#  main.py
#  imgproc_python
#
#  Created by Marco Garduno on 24/07/17.
#  Copyright 2017 Marco Garduno. All rights reserved.
#

import cv2
import functions as f
import morphology as m
import time

KERNEL_SIZE = 3

def main():
    start_time = time.time()

    image = cv2.imread("images/lena.png")
    if image is None:
        raise FileNotFoundError("Could not load image: images/lena.png")
    image_gray = f.rgb2gray(image)

    filtered = m.erosion(image_gray, KERNEL_SIZE)

    print("--- %s seconds ---" % (time.time() - start_time))
    cv2.imwrite("images/lena_eroded.png", filtered)

if __name__ == "__main__":
    main()
