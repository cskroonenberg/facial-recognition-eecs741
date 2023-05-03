import cv2
import math
import numpy as np
from hamming import HammingDistanceScoreMatrix
from system_b import SystemBScoreMatrix
from plot_matrix import plot_matrix

def DecidabilityIndex(score_matrix):
    genuine_scores = np.array([])
    imposter_scores = np.array([])
    for i in range(100):
        genuine_scores = np.append(genuine_scores, score_matrix[i, i])
        imposter_scores = np.append(imposter_scores, score_matrix[i, :i])
        imposter_scores = np.append(imposter_scores, score_matrix[i, (i+1):])

    genuine_mean = np.mean(genuine_scores)
    imposter_mean = np.mean(imposter_scores)

    genuine_std_dev = np.std(genuine_scores)
    imposter_std_dev = np.std(imposter_scores)

    num = math.sqrt(2)*abs(genuine_mean - imposter_mean)
    denom = math.sqrt(math.pow(genuine_std_dev, 2) + math.pow(imposter_std_dev, 2))

    return num/denom

def main():
    # Read in images
    gallery_imgs = []
    probe_imgs = []
    for i in range(100):
        gallery_img_gray = cv2.imread("data/gallery/subject{}_img1.pgm".format(i+1), cv2.IMREAD_GRAYSCALE)
        probe_img_gray = cv2.imread("data/probe/subject{}_img2.pgm".format(i+1), cv2.IMREAD_GRAYSCALE)
        gallery_imgs.append(gallery_img_gray)
        probe_imgs.append(probe_img_gray)

    # Generate score matrix for first facial recognition method
    A = HammingDistanceScoreMatrix(gallery_imgs.copy(), probe_imgs.copy())
    print("Part 1A: A[0:9, 0:9] snippet:")
    print(A[0:9, 0:9])

    hamming_decidability_idx = DecidabilityIndex(A)
    print("Part 1B: Hamming distance decidability index: {}".format(hamming_decidability_idx))

    # Generate score matrix for second facial recognition method
    B = SystemBScoreMatrix(gallery_imgs.copy(), probe_imgs.copy())
    print("Part 2A: B[0:9, 0:9] snippet:")
    print(B[0:9, 0:9])

    system_b_decidability_idx = DecidabilityIndex(B)
    print("Part 2B: System B decidability index: {}".format(system_b_decidability_idx))

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
    print("Part 2C Improvement Factor (IF) = {}\tScore = {}".format(improvement_factor, score))

if __name__ == "__main__":
    main()