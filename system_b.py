import cv2
import numpy as np
import dist

def System_B_Score_Matrix(gallery_imgs, probe_imgs):
    # cv2.imshow('original gallery', gallery_imgs[1])
    # cv2.imshow('original probe', probe_imgs[1])

    # Crop mouth
    filt_2d_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    for i, img in enumerate(gallery_imgs):
        gallery_imgs[i] = cv2.filter2D(gallery_imgs[i].copy(), ddepth= -1, kernel= filt_2d_kernel)
        gallery_imgs[i] = gallery_imgs[i][0:33, :]
        gallery_imgs[i] = cv2.GaussianBlur(gallery_imgs[i].copy(), (3,3), 0)
    for i, img in enumerate(probe_imgs):
        probe_imgs[i] = cv2.filter2D(probe_imgs[i].copy(), ddepth= -1, kernel= filt_2d_kernel)
        probe_imgs[i] = probe_imgs[i][0:33, :]
        probe_imgs[i] = cv2.GaussianBlur(probe_imgs[i].copy(), (3,3), 0)


    #cv2.imwrite('modified_gallery.png', gallery_imgs[1])
    #cv2.imwrite('modified_probe.png', probe_imgs[1])

    # Convert images to binary
    for i, img in enumerate(gallery_imgs):
        _, gallery_imgs[i] = cv2.threshold(img,60,255,cv2.THRESH_BINARY)
    for i, img in enumerate(probe_imgs):
        _, probe_imgs[i] = cv2.threshold(img,60,255,cv2.THRESH_BINARY)

    #cv2.imwrite('binarized_gallery.png', gallery_imgs[1])
    #cv2.imwrite('binarized_probe.png', probe_imgs[1])

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Calculate distance scores between gallery and probe images
    A = np.empty((100,100))
    for i in range(100):
        for j in range(100):
            A[i, j] = dist.EuclideanDistance(probe_imgs[i], gallery_imgs[j])
    
    return A