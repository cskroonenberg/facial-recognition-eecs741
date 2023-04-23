import cv2
import numpy as np

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

def TODO_DistanceScore(gallery_imgs, probe_imgs):
    # Convert images to binary
    for i, img in enumerate(gallery_imgs):
        _, gallery_imgs[i] = cv2.threshold(img,128,255,cv2.THRESH_BINARY)
    for i, img in enumerate(probe_imgs):
        _, probe_imgs[i] = cv2.threshold(img,128,255,cv2.THRESH_BINARY)

    # TODO: Check that the above binarization sets pixel values to 1 and 0 (instead of 255 and 0)

    B = np.empty((100,100))
    for i in range(100):
        for j in range(100):
            B[i, j] = EuclideanDistance(probe_imgs[i], gallery_imgs[j])
    
    return B