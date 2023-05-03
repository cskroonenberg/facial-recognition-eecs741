# Analyzing a Light-weight Face Recognition System with Further Improvement
## Overview
An analysis of the performance of a facial recognition biometric systems by considering two different feature binarization techniques. Two sets of images are provided (in ```data```): one "gallery" set, the database which contains the face templates of genuine users collected during enrollment phase and one "probe" set, face templates which are collected from the individuals who attempt to access the system as returning “users”. Each set contains 100 normalized grayscale face images of individuals (Probe Image 1 is face of the individual in Gallery Image 1).

The goal is to create a system which processes the images in a way that  distance function finds minimal distances between faces of the same person and maximal scores between faces of different people.

System performance is calculated based on the Decidability Index value, given by the following formula, defined in ```main.py``` as ```DecidabilityIndex(score_matrix)```:

<figure>
<img src=doc/decidabilityIndex.png>
</figure>

In this formula, ${\mu}_1$ and ${\mu}_2$ are the mean of the genuine and imposter score distribution respectively. Similarly, ${\sigma}_2$ and ${\sigma}_2$ denote the standard deviation of the genuine and impostor distribution respectively. Genuine scores are the distance scores of images with the same individual while imposter scores are the distance scores of images with different individuals.

A higher decidability index implies better performance. Systems with a higher decidability index value are better at differentiating the matching face from other faces.

## System A
#### Method
System A applies threshold binarization using ```cv2.threshold(img,128,1,cv2.THRESH_BINARY)``` to all gallery and probe images before calculating distances using Hamming distance, defined in ```dist.py``` as ```HammingDistance(img_a, img_b)```.

Gallery and probe images of the individual after threshold binarization:

<figure>
<img src=doc/sysA_binarized_gallery.png width="45%" height="45%">
<img src=doc/sysA_binarized_probe.png width="45%" height="45%">
</figure><br>

#### Results
System A yielded a Decidability Index value of ~2.9104. The score matrices are plotted below. Note that the scores of images with matching individuals are lower than the scores of images with different individuals.
<figure>
<img src=doc/System_A_full.png width="100%" height="100%">
<br>
<img src=doc/System_A_snippet.png width="100%" height="100%">
</figure><br>

## System B
#### Method
System B applies two pre-processing measures, ```cv2.filter2D``` and ```cv2.GaussianBlur```, and also crops the lower third of the images. Binarization is done using ```cv2.threshold(img,60,1,cv2.THRESH_BINARY)```. Finally, distance is calculated using Euclidean distance, defined in ```dist.py``` as ```EuclideanDistance(img_a, img_b)```. 

Gallery and probe images of the individual after pre-processing:
<figure>
<img src=doc/sysB_modified_gallery.png width="45%" height="45%">
<img src=doc/sysB_modified_probe.png width="45%" height="45%">
</figure><br>

Gallery and probe images of the individual after threshold binarization:
<figure>
<img src=doc/sysB_binarized_gallery.png width="45%" height="45%">
<img src=doc/sysB_binarized_probe.png width="45%" height="45%">
</figure><br>

#### Results
System B yielded a Decidability Index value of ~3.859. The score matrices are plotted below. Similar to System A, scores of images with matching individuals are lower than the scores of images with different individuals. However, note the consistency with which System B distinguishes between different individuals (hence the improved Decidability Index).
<figure>
<img src=doc/System_B_full.png width="100%" height="100%">
<br>
<img src=doc/System_B_snippet.png width="100%" height="100%">
</figure><br>