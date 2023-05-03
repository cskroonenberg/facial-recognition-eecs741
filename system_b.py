import cv2
import numpy as np
from hamming import HammingDistance

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

def LBP_Preprocess(img_set):
    for k, img in enumerate(img_set):
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
         img_set[k] = img
    return img_set

def TODO_DistanceScore(gallery_imgs, probe_imgs):
    gallery_imgs = cv2.erode(np.array(gallery_imgs), (3,3))
    probe_imgs = cv2.erode(np.array(probe_imgs), (3,3))
    gallery_imgs = cv2.GaussianBlur(np.array(gallery_imgs), (5,5), 0)
    probe_imgs = cv2.GaussianBlur(np.array(probe_imgs), (5,5), 0)
    gallery_imgs = cv2.medianBlur(np.array(gallery_imgs),3)
    probe_imgs = cv2.medianBlur(np.array(probe_imgs), 3)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    gallery_imgs = cv2.filter2D(np.array(gallery_imgs), ddepth= -1, kernel= kernel)
    probe_imgs = cv2.filter2D(np.array(probe_imgs), ddepth= -1, kernel=kernel)
    
    gallery_imgs = cv2.boxFilter(gallery_imgs.copy(), -1, (2,2), False)
    probe_imgs = cv2.boxFilter(probe_imgs.copy(), -1, (2,2), False)

    sift = cv2.SIFT_create(contrastThreshold=0.001, edgeThreshold=15, sigma=2.4)
    for i in range(len(gallery_imgs)):
        #keypoints, descriptors = orb.detectAndCompute(gallery_imgs[i], None)
        # find the keypoints with ORB
        keypoints = sift.detect(gallery_imgs[i],None)
        # compute the descriptors with ORB
        keypoints, des = sift.compute(gallery_imgs[i], keypoints)
        for kp in keypoints:
            x, y = kp.pt
            x, y = int(x), int(y)
            gallery_imgs[i][y][x] = 255
            if y < 99:
                gallery_imgs[i][y+1][x] = 255
            if y < 98:
                gallery_imgs[i][y+2][x] = 255
            if y > 0:
                gallery_imgs[i][y-1][x] = 255
            if y > 1:
                gallery_imgs[i][y-2][x] = 255
            if x < 99:
                gallery_imgs[i][y][x+1] = 255
            if x < 98:
                gallery_imgs[i][y][x+2] = 255
            if x > 0:
                gallery_imgs[i][y][x-1] = 255
            if x > 1:
                gallery_imgs[i][y][x-2] = 255

    for i in range(len(probe_imgs)):
        #keypoints, descriptors = orb.detectAndCompute(probe_imgs[i], None)
        # find the keypoints with ORB
        keypoints = sift.detect(probe_imgs[i],None)
        # compute the descriptors with ORB
        keypoints, des = sift.compute(probe_imgs[i], keypoints)
        for kp in keypoints:
            x, y = kp.pt
            x, y = int(x), int(y)
            probe_imgs[i][y][x] = 255
            if y < 99:
                probe_imgs[i][y+1][x] = 255
            if y < 98:
                probe_imgs[i][y+2][x] = 255
            if y > 0:
                probe_imgs[i][y-1][x] = 255
            if y > 1:
                probe_imgs[i][y-2][x] = 255
            if x < 99:
                probe_imgs[i][y][x+1] = 255
            if x < 98:
                probe_imgs[i][y][x+2] = 255
            if x > 0:
                probe_imgs[i][y][x-1] = 255
            if x > 1:
                probe_imgs[i][y][x-2] = 255

    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    # matched = bf.match([descriptors1],[descriptors2])
    
    cv2.imwrite("test_gallery.png", gallery_imgs[2])
    cv2.imwrite("test_probe.png", probe_imgs[2])

    B = np.empty((100,100))
    for i in range(100):
         for j in range(100):
            B[i, j] = EuclideanDistance(gallery_imgs[i], probe_imgs[j])
    
    return B