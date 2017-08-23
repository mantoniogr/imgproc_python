#!/usr/bin/env

#
#  functions.py
#  imgproc_python
#
#  Created by Marco Garduno on 24/07/17.
#  Copyright 2017 Marco Garduno. All rights reserved.
#

import cv2
import numpy as np

def rgb2gray(image):
    height, width, channels =  image.shape
    img = np.zeros([height, width], dtype=np.uint8)

    for j in range(0, height):
        for i in range(0, width):
            img[j,i] = 0.299*image[j,i,2] + 0.587*image[j,i,1] +\
             0.114*image[j,i,0]

    return img

def negative_gray(image):
    height, width =  image.shape
    img = np.copy(image)

    for j in range(0, height):
        for i in  range(0, width):
            img[j, i] = 255 - image[j, i]

    return img

def negative_color(image):
    height, width, channels =  image.shape
    img = np.copy(image)

    for j in range(0, height):
        for i in  range(0, width):
            img[j, i, 2] = 255 - image[j, i, 2]
            img[j, i, 1] = 255 - image[j, i, 1]
            img[j, i, 0] = 255 - image[j, i, 0]

    return img

def threshold_1(image, th1):
    height, width =  image.shape
    img = np.copy(image)

    for j in range(0, height):
        for i in  range(0, width):
            if image[j,i] > th1:
                img[j,i] = 255;
            else:
                img[j,i] = 0;

    return img

def threshold_2(image, th1, th2):
    height, width =  image.shape
    img = np.copy(image)

    for j in range(0, height):
        for i in  range(0, width):
            if image[j,i] > th1 and th2 > image[j,i]:
                img[j,i] = 255;
            else:
                img[j,i] = 0;

    return img

def counting_objects(image):
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
