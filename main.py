import cv2
import numpy as np

image = cv2.imread("images/lena.png")

cv2.imshow("Test", image)
cv2.waitKey(0)
