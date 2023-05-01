import cv2
import numpy as np

def HammingDistance(img_a, img_b):
    # TODO: Calculate Hamming Distance
    # COMPLETE(Javante): calculate hamming distance
    return len((img_a != img_b).nonzero()[0]) 

def HammingDistanceScoreMatrix(gallery_imgs, probe_imgs):
    # Convert images to binary
    for i, img in enumerate(gallery_imgs):
        _, gallery_imgs[i] = cv2.threshold(img,128,1,cv2.THRESH_BINARY)
    for i, img in enumerate(probe_imgs):
        _, probe_imgs[i] = cv2.threshold(img,128,1,cv2.THRESH_BINARY)
    
    A = np.empty((100,100))
    for i in range(100):
        for j in range(100):
            A[i, j] = HammingDistance(probe_imgs[i], gallery_imgs[j])
    
    return A