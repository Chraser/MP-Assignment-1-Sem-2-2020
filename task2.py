import cv2 as cv
import numpy as np
import sys
from pathlib import Path
from matplotlib import pyplot as plt
import os


'''
Contains code copied and adapted from 
https://stackoverflow.com/questions/6090399/get-hog-image-features-from-opencv-python/31042366#31042366
Last accessed on 03/10/2020
'''
def hog(image1, image2, fileName, fileExt, folder):
    gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    winSize = (64,128)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    hog = cv.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)

    winStride = (8,8)
    padding = (8,8)
    locations = ((10,20),)
    h1 = hog.compute(gray1,winStride,padding,locations)
    h2 = hog.compute(gray2,winStride,padding,locations)
    diff = cv.norm(h1-h2)
    original = cv.norm(h1)
    if(original > 0):
        ratio = diff/original*100
    else:
        ratio = diff
    print("HOG: " + str(ratio) + "%")

'''
Contains code copied and adapted from https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html 
and https://docs.opencv.org/3.4.2/da/df5/tutorial_py_sift_intro.html
Last accessed on 03/10/2020
'''
def siftCompare(image1, image2, fileName, fileExt, folder):
    gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    
    siftObj = cv.xfeatures2d.SIFT_create()
    kp1, des1 = siftObj.detectAndCompute(gray1,None)
    kp2, des2 = siftObj.detectAndCompute(gray2,None)

    bf = cv.BFMatcher()
    matches = bf.match(des1,des2)
    # Sort the matches to get the best match to ensure that we get the same keypoint from the 2 images
    matches = sorted(matches, key = lambda x:x.distance)
    result = cv.drawMatches(image1,kp1,image2,kp2,matches[:1], None, flags=2)
    cv.imwrite(folder + fileName + "-sift-matches" + fileExt, result)

    # Adjust ratio for cropping scaled images
    if("scale" in fileName):
        ratio = int(fileName.split("-")[2])
        offset = int(25 * (ratio / 100))
    else:
        offset = 25

    # Pick the best match that you can crop a square from the center of the keypoint
    index = 0
    for i in range (0,len(matches)):
        x1,y1 = [int(j) for j in kp1[matches[i].queryIdx].pt]
        x2,y2 = [int(j) for j in kp2[matches[i].trainIdx].pt]
        if(x1 > offset and y1 > offset):
            index = i
            break

    cropped1 = image1[y1-25:y1+25, x1-25:x1+25]
    cropped2 = image2[y2-offset:y2+offset, x2-offset:x2+offset]
    var1 = des1[matches[index].queryIdx]
    var2 = des2[matches[index].trainIdx]
    diff = cv.norm(var1 - var2)
    original = cv.norm(var1)
    print("SIFT variation: " + str(diff/original*100) + "%")
    return cropped1.copy(), cropped2.copy()

def main():
    if(len(sys.argv) == 3):
        path = Path(sys.argv[2])
        folder = "task2/"
        if not os.path.exists(folder):
            os.mkdir(folder)
        file = sys.argv[1]
        image1 = cv.imread(file)
        file2 = sys.argv[2]
        image2 = cv.imread(file2)
        cropped1, cropped2 = siftCompare(image1.copy(), image2.copy(), path.stem, path.suffix, folder)
        cv.imwrite(folder + path.stem + "-cropped1" + path.suffix, cropped1)
        cv.imwrite(folder + path.stem + "-cropped2" + path.suffix, cropped2)
        print("Keypoint cropped image ", end='')
        hog(cropped1.copy(), cropped2.copy(), path.stem, path.suffix, folder)
        print("Full image ", end='')
        hog(image1.copy(), image2.copy(), path.stem, path.suffix, folder)
    else:
        print("Please run the program with 'python task2.py file file2'")

if(__name__ == '__main__'):
    main()