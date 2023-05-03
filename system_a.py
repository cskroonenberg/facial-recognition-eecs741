import cv2
import numpy as np
import dist

def System_A_Score_Matrix(gallery_imgs, probe_imgs):
    # Convert images to binary
    for i, img in enumerate(gallery_imgs):
        _, gallery_imgs[i] = cv2.threshold(img,128,1,cv2.THRESH_BINARY)
    for i, img in enumerate(probe_imgs):
        _, probe_imgs[i] = cv2.threshold(img,128,1,cv2.THRESH_BINARY)
    
    A = np.empty((100,100))
    for i in range(100):
        for j in range(100):
            A[i, j] = dist.HammingDistance(probe_imgs[i], gallery_imgs[j])
    
    return A