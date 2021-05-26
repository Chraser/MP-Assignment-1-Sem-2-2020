import cv2 as cv
import numpy as np
import sys
from pathlib import Path

'''
Contains code copied from https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c/37347070#37347070
Last accessed on 30/09/2020
'''
def rotate(image, fileName, fileExt, angle):
    height,width = image.shape[:2]
    center = (width/2, height/2)
    rot = cv.getRotationMatrix2D(center, angle, 1.0)

    abs_cos = abs(rot[0,0])
    abs_sin = abs(rot[0,1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rot[0, 2] += (bound_w/2) - center[0]
    rot[1, 2] += (bound_h/2) - center[1]

    rotated = cv.warpAffine(image, rot, (bound_w, bound_h))

    cv.imwrite(fileName + "-rotated-by-" + str(angle) + fileExt, rotated)

def main():
    if(len(sys.argv) == 3):
        path = Path(sys.argv[1])
        file = sys.argv[1]
        image = cv.imread(file)
        angle = int(sys.argv[2])
        rotate(image, path.stem, path.suffix, angle)
    else:
        print("Please run the program with 'python rotate.py file scale'")

if(__name__ == '__main__'):
    main()