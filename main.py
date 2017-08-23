#!/usr/bin/env

#
#  main.py
#  imgproc_python
#
#  Created by Marco Garduno on 24/07/17.
#  Copyright 2017 Marco Garduno. All rights reserved.
#

import cv2
import numpy as np
import functions as f
import morphology as m
import time

start_time = time.time()

image = cv2.imread("images/objetos.bmp")
image_gray = f.rgb2gray(image)

filtered = m.minimos(image_gray)

cv2.imshow("Test", image)
cv2.imshow("Filtered", filtered)

print("--- %s seconds ---" % (time.time() - start_time))

cv2.waitKey(0)
