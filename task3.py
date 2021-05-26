import cv2 as cv
import numpy as np
import sys
from pathlib import Path
import os

'''
Contains code copied and adapted from https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html 
Last accessed on 30/09/2020
'''
def connectedComponentLabelling(image, fileName, fileExt, folder):
    # Best result when only using the red channel colours for Dugong image
    if(fileName.startswith("Dugong")):
        # Sets the blue colour channel to zero
        image[:,:,0] = 0
        # Sets the green colour channel to zero
        image[:,:,1] = 0
        # Median blur gets a 'thinner' binary image
        #image = cv.medianBlur(image,5)
        image = cv.GaussianBlur(image,(7,7),0)
        
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imwrite(fileName + '-gray' + fileExt, gray)
    if(fileName.startswith("Dugong")):
        (thresh, binary) = cv.threshold(gray, 128, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    else:
        # Use binary inv because the card's background is white
        (thresh, binary) = cv.threshold(gray, 128, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    numLabels, labels, stats, centroids = cv.connectedComponentsWithStats(binary, 8, cv.CV_32S)
    textFile = open(folder  +fileName+"-ccl-stats.txt","w")
    # numLabels - 1 because excluding the background
    textFile.write("Number of components: " + str(numLabels-1) + "\n")
    for i in range(1, numLabels):
        mask =  image.copy()
        mask[labels==i] = 0
        mask[labels!=i] = 255
        x, y, xOffset, yOffset, area= stats[i]
        crop = mask[y:y+yOffset, x:x+xOffset]
        cv.imwrite(folder + fileName + '-ccl-' + str(i) + fileExt, crop)
        textFile.write("Area of component " + str(i) + ": " +  str(area) + "\n")
    
    cv.imwrite(folder + fileName + '-binary' + fileExt, binary)

def main():
    if(len(sys.argv) == 2):
        path = Path(sys.argv[1])
        folder = "task3/"
        if not os.path.exists(folder):
            os.mkdir(folder)
        file = sys.argv[1]
        image = cv.imread(file)
        connectedComponentLabelling(image.copy(), path.stem, path.suffix, folder)
    else:
        print("Please run the program with 'python task3.py file'")

if(__name__ == '__main__'):
    main()