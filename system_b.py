import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

def EuclideanDistance(img_a, img_b):
    # TODO: Calculate distance between images using some distance function
    #COMPLETE(Michael): Calculate total Euclidean distance
    distance = 0
    for i in range(len(img_a)):
        distance += np.sqrt(np.sum((img_a[i]-img_b[i])**2))
    return distance

def CityBlockDistance(img_a, img_b):
    # TODO: Calculate distance between images using some distance function
    #COMPLETE(Michael): Calculate City Block distance
    distance = 0
    for i in range(len(img_a)):
        distance += np.sum(np.abs(img_a[i] - img_b[i]))
    return distance

def ChessboardDistance(img_a, img_b):
    # TODO: Calculate distance between images using some distance function
    #Incomplete(Michael):Calculate Chessboard distance
    distance = 0
    for i in range(len(img_a)):
        distance += np.max(np.abs(img_a[i] - img_b[i]))
    return distance
    #return np.max(np.abs(img_a - img_b))

def HammingDistance(img_a, img_b):
    # TODO: Calculate Hamming Distance
    # COMPLETE(Javante): calculate hamming distance
    return len((img_a != img_b).nonzero()[0])

def TODO_DistanceScore(gallery_imgs, probe_imgs):
    # Convert images to binary
    # for i, img in enumerate(gallery_imgs):
    #     _, gallery_imgs[i] = cv2.threshold(img,128,255,cv2.THRESH_BINARY)
    # for i, img in enumerate(probe_imgs):
    #     _, probe_imgs[i] = cv2.threshold(img,128,255,cv2.THRESH_BINARY)

    # TODO: Check that the above binarization sets pixel values to 1 and 0 (instead of 255 and 0)

    #LBP Pre-processing
    for k, img in enumerate(gallery_imgs):
         img_length = len(img) - 2
         img_width = len(img[0]) - 2
         lbp_img = np.zeros((img_length, img_width))
         for i in range(1, img_length):
             for j in range(1, img_width):
                 center = (i+1, j+1)
                 center_value = img[center[0]][center[1]]
                 bit = 1
                 bit_value = 0
                 for z in range(3):
                     for t in range(3):
                         bit_value += bit if img[z+i][t+j] >= center_value else 0
                         bit *= 2
                 img[center[0]][center[1]] = bit_value
         gallery_imgs[k] = img

    for k, img in enumerate(probe_imgs):
        img_length = len(img) - 2
        img_width = len(img[0]) - 2
        lbp_img = np.zeros((img_length, img_width))
        for i in range(1, img_length):
            for j in range(1, img_width):
                center = (i+1, j+1)
                center_value = img[center[0]][center[1]]
                bit = 1
                bit_value = 0
                for z in range(3):
                    for t in range(3):
                        bit_value += bit if img[z+i][t+j] >= center_value else 0
                        bit *= 2
                img[center[0]][center[1]] = bit_value
        probe_imgs[k] = img

    #Apply more pre-processing and binarize image
    sigma = 0.2
    apt = 7
    l2 = True
    for i, img in enumerate(gallery_imgs):
        img =  cv2.medianBlur(img,5)
        #img = cv2.GaussianBlur(img,(5,5),0)
        med = np.median(img)
        #lwr = int(max(0, (1.0 - (sigma)) * med))
        #upr = int(min(255, (1.0 + (sigma)) * med))
        lwr = 50
        upr = 200
        #gallery_imgs[i] = cv2.Canny(img, lwr, upr, 5, apertureSize=apt, L2gradient=l2) #Canny
        #_, gallery_imgs[i] = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #OTSU 
        gallery_imgs[i] = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY , 13, 8) #Adaptive
    for i, img in enumerate(probe_imgs):
        img = cv2.medianBlur(img,5)
        #img = cv2.GaussianBlur(img,(5,5),0)
        med = np.median(img)
        lwr = int(max(0, (1.0 - sigma) * med))
        upr = int(min(255, (1.0 + sigma) * med))
        #sprobe_imgs[i] = cv2.Canny(img, lwr, upr, apertureSize=apt, L2gradient=l2) #Canny
        #_, probe_imgs[i] = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #OTSU threshold
        probe_imgs[i] = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY , 13, 8) #Adaptive

    #Convert to binary
    for k, img in enumerate(gallery_imgs):
        for i in range(len(img)):
            for j in range(len(img[0])):
                if img[i][j] == 255:
                    img[i][j] = 1
        gallery_imgs[k] = img

    for k, img in enumerate(probe_imgs):
        for i in range(len(img)):
            for j in range(len(img[0])):
                if img[i][j] == 255:
                    img[i][j] = 1
        probe_imgs[k] = img

    B = np.empty((100,100))
    for i in range(100):
        for j in range(100):
            B[i, j] = EuclideanDistance(probe_imgs[i], gallery_imgs[j])
    
    return B