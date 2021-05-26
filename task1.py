import cv2 as cv
import numpy as np
import sys
from pathlib import Path
from matplotlib import pyplot as plt
import os

'''
Contains code copied and adapted from https://docs.opencv.org/3.4.2/d1/db7/tutorial_py_histogram_begins.html
Last accessed on 03/10/2020
'''
def histogram(image, fileName, fileExt, folder):
    color = ('b','g','r')
    for i,j in enumerate(color):
        plt.hist(image[:,:,i].reshape(-1), 10, color=j)
        #plt.hist(image[:,:,i].ravel(), 10, color=j)
        plt.xlabel('Colour intensity')
        plt.ylabel('Pixel frequency')
        plt.xlim(0,255)
        plt.savefig(folder + fileName + "-histogram-" + j + fileExt)
        plt.clf()

'''
Contains code copied and adapted from https://docs.opencv.org/3.4.2/dc/d0d/tutorial_py_features_harris.html
Last accessed on 03/10/2020
'''
def harrisDetection(image, fileName, fileExt, folder):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    dst = cv.dilate(dst, None)
    image[dst>0.01*dst.max()] = [255,0,0]
    cv.imwrite(folder + fileName + "-harris" + fileExt, image)

'''
Contains code copied and adapted from https://docs.opencv.org/3.4.2/da/df5/tutorial_py_sift_intro.html
Last accessed on 03/10/2020
'''
def sift(image, fileName, fileExt, folder):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    siftObj = cv.xfeatures2d.SIFT_create()
    kp, des = siftObj.detectAndCompute(gray,None)
    textFile = open(folder + fileName + "-sift-descriptors.txt","w")
    np.set_printoptions(threshold=np.inf)
    textFile.write(np.array_str(des))
    image = cv.drawKeypoints(gray, kp, image)
    cv.imwrite(folder + fileName + "-sift-keypoints" + fileExt, image)
    cv.imwrite(folder + fileName + "-sift-descriptors" + fileExt, des)

def main():
    if(len(sys.argv) == 2):
        path = Path(sys.argv[1])
        folder = "task1/"
        if not os.path.exists(folder):
            os.mkdir(folder)
        file = sys.argv[1]
        image = cv.imread(file)
        if(path.stem.startswith("Dugong")):
            image = cv.GaussianBlur(image, (7,7),0)
            #image = cv.medianBlur(image, 5)
        histogram(image.copy(), path.stem, path.suffix, folder)
        harrisDetection(image.copy(), path.stem, path.suffix, folder)
        sift(image.copy(), path.stem, path.suffix, folder)

        
    else:
        print("Please run the program with 'python task1.py file'")

if(__name__ == '__main__'):
    main()