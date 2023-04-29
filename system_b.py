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
    #return np.max(np.abs(img_a - img_b))

# def TODO_DistanceScore(gallery_imgs, probe_imgs):
#     # Convert images to binary
#     for i, img in enumerate(gallery_imgs):
#         _, gallery_imgs[i] = cv2.threshold(img,128,255,cv2.THRESH_BINARY)
#     for i, img in enumerate(probe_imgs):
#         _, probe_imgs[i] = cv2.threshold(img,128,255,cv2.THRESH_BINARY)

#     # TODO: Check that the above binarization sets pixel values to 1 and 0 (instead of 255 and 0)

#     B = np.empty((100,100))
#     for i in range(100):
#         for j in range(100):
#             B[i, j] = EuclideanDistance(probe_imgs[i], gallery_imgs[j])
    
#     return B

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
    gallery_imgs = cv2.GaussianBlur(np.array(gallery_imgs), (5,5), 0)
    probe_imgs = cv2.GaussianBlur(np.array(probe_imgs), (5,5), 0)
    gallery_imgs = cv2.medianBlur(np.array(gallery_imgs),5)
    probe_imgs = cv2.medianBlur(np.array(probe_imgs), 5)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    # kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    gallery_imgs = cv2.filter2D(np.array(gallery_imgs), ddepth= -1, kernel= kernel)
    probe_imgs = cv2.filter2D(np.array(probe_imgs), ddepth= -1, kernel=kernel)
    
    orb = cv2.ORB_create()
    for img in gallery_imgs:
        keypoints1, descriptors1 = orb.detectAndCompute(img, None)
    print("we also live maybe")
    for img in probe_imgs:  
        keypoints2, descriptors2 = orb.detectAndCompute(img, None)
    print("we also live")

# Different features tested, LBP,SIFT,ORB, Binarization by mean, Binarization by thresholding 
#This implementation give the exact same scare and hamming distance index like ORB so yeah
    # sift = cv2.SIFT_create()
    # for img in gallery_imgs:
    #      keypoints1 = sift.detect(img, None)
    
    # for img in probe_imgs:
    #      keypoints2 = sift.detect(img, None)
    
    for kp in [keypoints1]:
        print("idk are we?")
        for i in range(len(kp)):
            x, y = kp[i].pt
            x, y = int(x), int(y)
            print("idk are we?")
            if np.all(gallery_imgs[y][x]) == 255:
                kp[i].response = 1
            else:
                kp[i].response = 0 
    print("idk are we?")
    for kp in [keypoints2]:
        for i in range(len(kp)):
            x, y = kp[i].pt
            x, y = int(x), int(y)

            if np.all(probe_imgs[y][x]) == 255:
                kp[i].response = 1
            else:
                kp[i].response = 0 

    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    # matched = bf.match([descriptors1],[descriptors2])

    B = np.empty((100,100))
    for i in range(100):
         for j in range(100):
             B[i, j] = EuclideanDistance(gallery_imgs[i], probe_imgs[j])
             #B[i, j] = EuclideanDistance(descriptors1[j.queryIdx], descriptors2[j.trainIdx])
    
    return B


#Perform the equalization of the images.. I am also aware that I could have done this in fewer lines of code
    #Equalizing gallery images
    # equ_gallery_images = []
    # equ_gallery = [cv2.equalizeHist(imgs) for imgs in gallery_imgs]
    # equ_gallery_images.append(equ_gallery)
    # print("Works till here")
    # #equalizing probe images
    # equ_probes_images = []
    # equ_probes = [cv2.equalizeHist(imgs) for imgs in probe_imgs]
    # equ_probes_images.append(equ_probes)
    # print("Works till here too")
    #  #performing the feature extraction on both data sets
    # orb = cv2.ORB_create()
    # galleryKeypoints = []
    # galleryDescriptors = []
    # probeKeypoints = []
    # probeDescriptors = []
    # equ_gallery_images = np.array(equ_gallery_images)
    # print("I work!")
    # for imgs in equ_gallery_images:
    #     print("Me too")
    #     Keypoints,Descriptors = orb.detectAndCompute(imgs,None)
    #     print("Not me")
    #     galleryKeypoints.append(Keypoints)
    #     galleryDescriptors.append(Descriptors)
    
    # for img in equ_probes_images:
    #     Keypoints,Descriptors = orb.detectAndCompute(img,None)
    #     probeKeypoints.append(Keypoints)
    #     probeDescriptors.append(Descriptors)

    # #binarization the equalized and extracted images

    # gallery_binary_array = []
    # probe_binary_array = []
    
    # for id in galleryDescriptors:
    #     gallery_bin_des = []
    #     for ids in id:
    #         bin_string = ''
    #         for num in ids:
    #             bin_string += bin(num)[2:].zfill(8)
    #         gallery_bin_des.append(bin_string)
    #     gallery_binary_array.append(gallery_bin_des)

    # gallery_binary_array = np.array(gallery_binary_array)

    # for id in probeDescriptors:
    #     probe_bin_des = []
    #     for ids in id:
    #         bin_string = ''
    #         for num in ids:
    #             bin_string += bin(num)[2:].zfill(8)
    #         probe_bin_des.append(bin_string)
    #     probe_binary_array.append(probe_bin_des)

    # probe_binary_array = np.array(probe_binary_array)

    # for i in range(len(gallery_imgs)):
    #     if len(gallery_imgs[i].shape) < 3:
    #         channels = 1
    #     else:
    #         channels = gallery_imgs[i].shape[2]
    #     print(f"Image {i} has {channels} channels.")
    #print ("We live")
    
    # gallery_imgs = LBP_Preprocess(gallery_imgs)
    # probe_imgs = LBP_Preprocess(probe_imgs)

    # #Equalizing gallery images
    # equ_gallery_images = []
    # equ_gallery = [cv2.equalizeHist(imgs) for imgs in gallery_imgs]
    # gallery_imgs = np.array(equ_gallery)

    # print("Works till here")
    # # #equalizing probe images
    # equ_probes_images = []
    # equ_probes = [cv2.equalizeHist(imgs) for imgs in np.array(probe_imgs)]
    # probe_imgs = np.array(equ_probes)
