import cv2
import numpy as np

def TODO_Distance(img_a, img_b):
    # TODO: Calculate distance between images using some distance function
    raise NotImplementedError

def TODO_DistanceScore(gallery_imgs, probe_imgs):
    # Convert images to binary
    for i, img in enumerate(gallery_imgs):
        gallery_imgs[i] = cv2.threshold(img,128,255,cv2.THRESH_BINARY)
    for i, img in enumerate(probe_imgs):
        probe_imgs[i] = cv2.threshold(img,128,255,cv2.THRESH_BINARY)

    # TODO: Check that the above binarization sets pixel values to 1 and 0 (instead of 255 and 0)

    B = np.empty((100,100))
    for i in range(100):
        for j in range(100):
            B[i, j] = TODO_Distance(probe_imgs[i], gallery_imgs[j])
    
    return B