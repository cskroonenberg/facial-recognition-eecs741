import cv2
import numpy as np

def HammingDistance(img_a, img_b):
    # TODO: Calculate Hamming Distance
    # COMPLETE(Javante): calculate hamming distance
    return len((img_a != img_b).nonzero()[0]) 

def HammingDistanceScoreMatrix(gallery_imgs, probe_imgs):
    # Convert images to binary
    for i, img in enumerate(gallery_imgs):
        gallery_imgs[i] = cv2.threshold(img,128,255,cv2.THRESH_BINARY)
    for i, img in enumerate(probe_imgs):
        probe_imgs[i] = cv2.threshold(img,128,255,cv2.THRESH_BINARY)

    # TODO: Check that the above binarization sets pixel values to 1 and 0 (instead of 255 and 0)
    # COMPLETE(Javante): convert arrays into 1s and 0s instead of 255s and 0s
    for k, x in enumerate(gallery_imgs):
        img = x[1]
        for i in range(len(img)):
            for j in range(len(img[0])):
                if img[i][j] == 255:
                    img[i][j] = 1
        gallery_imgs[k] = img

    for k, x in enumerate(probe_imgs):
        img = x[1]
        for i in range(len(img)):
            for j in range(len(img[0])):
                if img[i][j] == 255:
                    img[i][j] = 1
        probe_imgs[k] = img
    
    A = np.empty((100,100))
    for i in range(100):
        for j in range(100):
            A[i, j] = HammingDistance(probe_imgs[i], gallery_imgs[j])
    
    return A