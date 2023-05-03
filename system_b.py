import cv2
import numpy as np
from hamming import HammingDistance

def SystemBScoreMatrix(gallery_imgs, probe_imgs):
    cv2.imshow('original gallery', gallery_imgs[1])
    cv2.imshow('original probe', probe_imgs[1])
    # Crop mouth
    for i, img in enumerate(gallery_imgs):
        gallery_imgs[i] = gallery_imgs[i][0:33, :]
    for i, img in enumerate(probe_imgs):
        probe_imgs[i] = probe_imgs[i][0:33, :]


    cv2.imshow('modified gallery', gallery_imgs[1])
    cv2.imshow('modified probe', probe_imgs[1])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convert images to binary
    for i, img in enumerate(gallery_imgs):
        _, gallery_imgs[i] = cv2.threshold(img,60,1,cv2.THRESH_BINARY)
    for i, img in enumerate(probe_imgs):
        _, probe_imgs[i] = cv2.threshold(img,60,1,cv2.THRESH_BINARY)

    A = np.empty((100,100))
    for i in range(100):
        for j in range(100):
            A[i, j] = HammingDistance(probe_imgs[i], gallery_imgs[j])
    
    return A