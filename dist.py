import numpy as np

def EuclideanDistance(img_a, img_b):
    # TODO: Calculate distance between images using some distance function
    #COMPLETE(Caden): Calculate total Euclidean distance
    distance = 0
    diff_arr = np.logical_xor(img_a, img_b)
    distance = diff_arr.sum()
    distance = distance**0.5
    return distance

def HammingDistance(img_a, img_b):
    # TODO: Calculate Hamming Distance
    # COMPLETE(Javante): calculate hamming distance
    return len((img_a != img_b).nonzero()[0])

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