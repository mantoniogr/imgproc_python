#!/usr/bin/env

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

start_time = time.time()

image = cv2.imread("images/lena.png")
image_gray = f.rgb2gray(image)

filtered = m.dilation(image_gray, 5)

cv2.imshow("Test", image)
cv2.imshow("Filtered", filtered)

print("--- %s seconds ---" % (time.time() - start_time))

cv2.waitKey(0)
