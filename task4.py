import cv2 as cv
import numpy as np
import sys
from pathlib import Path
from matplotlib import pyplot as plt
import os

def method_red_only(image):
    # Good image segmentation result if only using the red channel 
    # colours for Dugong image but doesnt work with diamond image
    # Sets the blue colour channel to zero
    image[:,:,0] = 0
    # Sets the green colour channel to zero
    image[:,:,1] = 0
    return image

def method_hsv_no_v(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # Sets the Value colour channel to zero
    image[:,:,2] = 0
    return image

'''
Contains code copied and adapted from https://docs.opencv.org/3.4.2/d1/d5c/tutorial_py_kmeans_opencv.html 
The code I've written below is similar to https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/ 
but I didn't copy the code from the website except adapted code on how to disable clusters to display a certain cltuer
and i did read the comments in the code in that website for further understanding of how kmeans image segmentation work in OpenCV
Last accessed on 30/09/2020
'''
def kMeans(original, image, _type, fileName, fileExt, folder):
    
    # reshape the image to a 2D array of dimension Mx3 where M is the number of pixels and 3 is number of colour channels in BGR
    pixel_values = image.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)

    # define stopping criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # number of clusters (K)
    k = 2
    # apply kmeans
    _, labels, (centers) = cv.kmeans(pixel_values, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values, only needed if want to show full segmented image
    centers = np.uint8(centers)
    
    # flatten the labels array
    labels = labels.flatten()
    textFile = open(folder + fileName+"-labels.txt","w")
    np.set_printoptions(threshold=np.inf)
    textFile.write(np.array_str(labels))

    # loop through all cluster and only output the image of that cluster
    for i in range(0,k):
        segmented_image = np.copy(original)
        # convert to 2d array similiar to what is used for k-means so that we can set clusters of pixels to have the pixel value of black colour
        segmented_image = segmented_image.reshape((-1, 3))
        # set every other cluster that isn't i to be black colour
        segmented_image[labels != i] = [0, 0, 0]
        # convert back to original shape
        segmented_image = segmented_image.reshape(image.shape)
        # save the image
        cv.imwrite(folder + fileName + "-" + _type + '-kmeans-' + str(i) + fileExt, segmented_image)

def main():
    if(len(sys.argv) == 2):
        path = Path(sys.argv[1])
        folder = "task4/"
        if not os.path.exists(folder):
            os.mkdir(folder)
        file = sys.argv[1]
        original = cv.imread(file)
        image = original.copy()
        kMeans(original, image, "BGR", path.stem, path.suffix, folder)
        image = method_red_only(original.copy())
        kMeans(original, image, "R", path.stem, path.suffix, folder)
        image = method_hsv_no_v(original.copy())
        kMeans(original, image, "HS", path.stem, path.suffix, folder)
    else:
        print("Please run the program with 'python task4.py file'")

if(__name__ == '__main__'):
    main()