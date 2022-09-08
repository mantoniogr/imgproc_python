#!/usr/bin/env

#
#  morphology.py
#  imgproc_python
#
#  Created by Marco Garduno on 24/07/17.
#  Copyright 2017 Marco Garduno. All rights reserved.
#

import cv2
import numpy as np
import functions as f

def dilation(map, size):
    height, width =  map.shape
    auxMap = np.copy(map)

    for k in range(0, size):
        # B1
        for j in range(0, height):
            for i in range(0, width-1):
                if auxMap[j, i] < auxMap[j, i+1]:
                    auxMap[j, i] = auxMap[j, i+1]

        # B2
        for j in range(0, height-1):
            for i in range(0, width):
                if auxMap[j, i] < auxMap[j+1, i]:
                    auxMap[j, i] = auxMap[j+1, i]

        # B3
        for j in range(0, height):
            for i in range(width-1, 0, -1):
                if auxMap[j, i] < auxMap[j, i-1]:
                    auxMap[j, i] = auxMap[j, i-1]

        # B4
        for j in range(height-1, 0, -1):
            for i in range(0, width):
                if auxMap[j, i] < auxMap[j-1, i]:
                    auxMap[j, i] = auxMap[j-1, i]

    return auxMap

def erosion(map, size):
    height, width =  map.shape
    auxMap = np.copy(map)

    auxMap = f.negative_gray(auxMap)
    auxMap = dilation(auxMap,size)
    auxMap = f.negative_gray(auxMap)

    return auxMap

def opening(map, size):
    auxMap = np.copy(map)

    auxMap = erosion(auxMap, size)
    auxMap = dilation(auxMap, size)

    return auxMap

def closing(map, size):
    auxMap = np.copy(map)

    auxMap = dilation(auxMap, size)
    auxMap = erosion(auxMap, size)

    return auxMap

def sequential_1(map, size):
    auxMap = np.copy(map)

    for i in range(0, size):
        auxMap = opening(auxMap, i+1)
        auxMap = closing(auxMap, i+1)

    return auxMap

def sequential_2(map, size):
    auxMap = np.copy(map)

    for i in range(0, size):
        auxMap = closing(auxMap, i+1)
        auxMap = opening(auxMap, i+1)

    return auxMap

def white_top_hat(map, size):
    height, width =  map.shape
    imgOriginal = np.copy(map)
    auxMap = np.copy(map)

    auxMap = opening(auxMap, size)

    for j in range(0, height):
        for i in range(0, width):
            auxMap[j, i] = imgOriginal[j, i] - auxMap[j, i]

    return auxMap

def black_top_hat(map, size):
    height, width =  map.shape
    imgOriginal = np.copy(map)
    auxMap = np.copy(map)

    auxMap = closing(auxMap, size)

    for j in range(0, height):
        for i in range(0, width):
            auxMap[j, i] = auxMap[j, i] - imgOriginal[j, i]

    return auxMap

def gradient(map, size):
    auxMap = np.copy(map)
    auxMap2 = np.copy(map)

    auxMap = dilation(auxMap, size)
    auxMap2 = erosion(auxMap2, size)

    auxMap = auxMap - auxMap2

    return auxMap

def external_gradient(map, n):
    auxMap = np.copy(map)
    imgOriginal = np.copy(map)

    auxMap = dilation(auxMap, n)
    auxMap = auxMap - imgOriginal

    return auxMap

def internal_gradient(map, n):
    auxMap = np.copy(map)
    imgOriginal = np.copy(map)

    auxMap = erosion(auxMap, n)
    auxMap = imgOriginal - auxMap

    return auxMap

def geodesic_dilation(I, J):
    height, width =  I.shape
    flag = True

    while(flag):

        img_auxiliar = np.copy(J)

        for j in range(1, height):
            for i in range(1, width-1):
                list1 = (   J[j-1, i-1], J[j-1, i],   J[j-1, i+1],
                            J[j, i-1],   J[j, i])
                J[j, i] = min([max(list1), I[j,i]])

        for j in range(height-2, -1, -1):
            for i in range(width-2, 0, -1):
                list2 = (                   J[j, i],     J[j, i+1],
                            J[j+1, i-1], J[j+1, i],   J[j+1, i+1] )
                J[j, i] = min([max(list2), I[j, i]])

        dif = J - img_auxiliar

        if np.amax(dif) == 0:
            flag = False

    return J

def geodesic_erosion(I, J):

    I = f.negative_gray(I)
    J = f.negative_gray(J)

    J = geodesic_dilation(I,J)

    I = f.negative_gray(I)
    J = f.negative_gray(J)

    return J

def opening_by_reconstruction(map, n):
    img_auxiliar = np.copy(map)
    Y = np.copy(map)

    Y = erosion(map, n)
    erosionada = np.copy(Y)
    J = geodesic_dilation(img_auxiliar, Y)

    return J

def closing_by_reconstruction(map, n):
    img_auxiliar = np.copy(map)
    Y = np.copy(map)

    Y = dilation(map, n)
    dilatada = np.copy(Y)
    J = geodesic_erosion(img_auxiliar, Y)

    return J

def sequential_reconstuction1(img, n):
    img_auxiliar = np.copy(img)
    for i in range(1, n+1):
        img_auxiliar = opening_by_reconstruction(img_auxiliar, i)
        img_auxiliar = closing_by_reconstruction(img_auxiliar, i)

    return img_auxiliar

def sequential_reconstruction2(img, n):
    img_auxiliar = np.copy(img)
    for i in range(1, n+1):
        img_auxiliar = closing_by_reconstruction(img_auxiliar, i)
        img_auxiliar = opening_by_reconstruction(img_auxiliar, i)

    return img_auxiliar

def maxima(img):
    height, width =  img.shape
    img_auxiliar = np.copy(img)

    for j in range(0, height):
        for i in range(0, width):
            if img_auxiliar[j,i] > 0:
                img_auxiliar[j,i] = img_auxiliar[j,i] - 1

    img_auxiliar = geodesic_dilation(img, img_auxiliar)
    img = img - img_auxiliar
    img = f.threshold_2(img, 1, 255)

    return img

def minimos(img):
    img_auxiliar = np.copy(img)

    img_auxiliar = f.negative_gray(img_auxiliar)
    img_auxiliar = maxima(img_auxiliar)
#   img_auxiliar = f.negativoGrises(img_auxiliar)

    return img_auxiliar

# Modified Closing by Reconstruction - Start

def minNoZero(lista):
    minimo = 255

    # for x in lista:
    #   if x < 255 and x > 0:
    #       minimo = x

    if lista[0] < 255 and lista[0] > 0:
        minimo = lista[0]
    if lista[1] < 255 and lista[1] > 0:
        minimo = lista[1]

    return minimo

def maxNo255(lista):
    maximo = 0
    for x in lista:
        if x > 0 and x < 255:
            maximo = x

    return maximo

def noiseCount(img):
    height, width, channels =  img.shape

    counter = 0

    for j in range(0,height):
        for i in range(0, width):
            if (img[j,i,0] == 0):
                counter = 1 + counter

    print(counter)
    return counter

def highPass(f, g):
    height, width, channels =  f.shape
    img_auxiliar = np.copy(f)

    for j in range(0, height):
        for i in range(0, width):
            # if (f[j,i,0] > g[j,i,0] and f[j,i,0] < 255):
            # if (f[j,i,0] < 255):
            if (f[j,i,0] > 0):
                img_auxiliar[j,i,0] = f[j,i,0]
            else:
                img_auxiliar[j,i,0] = g[j,i,0]

    img_auxiliar[:,:,1] = img_auxiliar[:,:,0]
    img_auxiliar[:,:,2] = img_auxiliar[:,:,0]

    return img_auxiliar

def dilatacionGeodesicaM(I, J):
    height, width =  I.shape
    flag = True

    while(flag):

        img_auxiliar = np.copy(J)

        for j in range(1, height):
            for i in range(1, width-1):
                list1 = (   J[j-1, i-1],     J[j-1, i],   J[j-1, i+1],
                            J[j, i-1],       J[j, i])
                J[j, i] = minNoZero([max(list1), I[j,i]])

        for j in range(height-2, -1, -1):
            for i in range(width-2, 0, -1):
                list2 = (                       J[j, i],     J[j, i+1],
                            J[j+1, i-1],     J[j+1, i],   J[j+1, i+1] )
                J[j, i] = minNoZero([max(list2), I[j, i]])

        dif = J - img_auxiliar

        if np.amax(dif) == 0:
            flag = False

    return J

def erosionGeodesicaM(I, J):

    I = negativo(I)
    J = negativo(J)

    J = dilatacionGeodesicaM(I,J)

    I = negativo(I)
    J = negativo(J)

    return J

def erosionGeodesicaM2(I, J):
    height, width, channels =  I.shape
    flag = True

    while(flag):

        img_auxiliar = np.copy(J)

        for j in range(1, height):
            for i in range(1, width-1):
                list1 = (   J[j-1, i-1, 0],     J[j-1, i, 0],   J[j-1, i+1, 0],
                            J[j, i-1, 0],       J[j, i, 0])
                J[j, i, 0] = maxNo255([min(list1), I[j,i,0]])

        for j in range(height-2, -1, -1):
            for i in range(width-2, 0, -1):
                list2 = (                       J[j, i, 0],     J[j, i+1, 0],
                            J[j+1, i-1, 0],     J[j+1, i, 0],   J[j+1, i+1, 0] )
                J[j, i, 0] = maxNo255([min(list2), I[j, i, 0]])

        dif = J - img_auxiliar

        if np.amax(dif) == 0:
            flag = False

    J[:,:,1] = J[:,:,0]
    J[:,:,2] = J[:,:,0]

    return J

def aperturaReconstruccionM(map, n):
    img_auxiliar = np.copy(map)
    Y = np.copy(map)

    Y = erosion(map, n)
    erosionada = np.copy(Y)
    J = dilatacionGeodesicaM(img_auxiliar, Y)

    return J, erosionada

def cerraduraReconstruccionM(map, n):
    img_auxiliar = np.copy(map)
    Y = np.copy(map)

    Y = dilation(map, n)
    dilatada = np.copy(Y)
    J = erosionGeodesicaM(img_auxiliar, Y)

    return J, dilatada

# Modified Closing by Reconstruction - End
