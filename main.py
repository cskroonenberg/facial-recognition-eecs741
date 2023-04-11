import cv2
import numpy as np
from hamming import HammingDistanceScore
from system_b import TODO_DistanceScore

def DecidabilityIndex(score_matrix):
    # TODO: Calculate decidability index
    raise NotImplementedError

def main():
    # Read in images
    gallery_imgs = []
    probe_imgs = []
    for i in range(100):
        gallery_img_gray = cv2.imread("data/gallery/subject{}_img1.pgm".format(i+1), cv2.IMREAD_GRAYSCALE)
        probe_img_gray = cv2.imread("data/probe/subject{}_img2.pgm".format(i+1), cv2.IMREAD_GRAYSCALE)
        gallery_imgs.append(gallery_img_gray)
        probe_imgs.append(probe_img_gray)

    assert len(gallery_imgs) == 100
    assert len(probe_imgs) == 100

    # Generate score matrix for first facial recognition method
    A = HammingDistanceScore(gallery_imgs, probe_imgs)
    print(A.shape)
    assert A.shape == (100, 100)

    print("A[0:9, 0:9] snippet:")
    print(A[0:9, 0:9])

    hamming_decidability_idx = DecidabilityIndex(A)

    # Generate score matrix for second facial recognition method
    B = TODO_DistanceScore(gallery_imgs, probe_imgs)
    assert B.shape == (100, 100)

    system_b_decidability_idx = DecidabilityIndex(B)

    improvement_factor = round((system_b_decidability_idx - hamming_decidability_idx), 2)
    if improvement_factor < 0:
        score = 0
    elif improvement_factor == 0:
        score = 5
    elif improvement_factor >= 0.1 and improvement_factor < 0.4:
        score = 10
    elif improvement_factor >= 0.4 and improvement_factor < 0.9:
        score = 15
    elif improvement_factor >= 0.9 and improvement_factor < 1.5:
        score = 18
    else:
        score = 20
    print("Part 2c score: {}".format(score))

if __name__ == "__main__":
    main()