import cv2
import numpy as np

def HammingDistance(img_a, img_b):
    # TODO: Calculate Hamming Distance
    raise NotImplementedError 

def HammingDistanceScoreMatrix(gallery_imgs, probe_imgs):
    # Convert images to binary
    for i, img in enumerate(gallery_imgs):
        gallery_imgs[i] = cv2.threshold(img,128,255,cv2.THRESH_BINARY)
    for i, img in enumerate(probe_imgs):
        probe_imgs[i] = cv2.threshold(img,128,255,cv2.THRESH_BINARY)

    # TODO: Check that the above binarization sets pixel values to 1 and 0 (instead of 255 and 0)
    
    A = np.empty((100,100))
    for i in range(100):
        for j in range(100):
            A[i, j] = HammingDistance(probe_imgs[i], gallery_imgs[j])
    
    return A

def HammingDecidability(score_matrix):
    # TODO: Calculate deicidability
    raise NotImplementedError 