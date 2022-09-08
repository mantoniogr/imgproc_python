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
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
logging.info('Running basic example')

start_time = time.time()

image = cv2.imread("images/lena.png")
image_gray = f.rgb2gray(image)

filtered = m.dilation(image_gray, 5)

print("--- %s seconds ---" % (time.time() - start_time))

cv2.imshow("Test", image)
cv2.imshow("Filtered", filtered)

cv2.waitKey(0)
