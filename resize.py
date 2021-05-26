import cv2 as cv
import numpy as np
import sys
from pathlib import Path

'''
Conains code copied and modified from https://docs.opencv.org/3.4.2/da/d6e/tutorial_py_geometric_transformations.html
Last accessed 30/9/2020
'''
def resize(image, fileName, fileExt, scale):
        resized = cv.resize(image, None, fx=scale/100, fy=scale/100, interpolation=cv.INTER_CUBIC)
        cv.imwrite(fileName + "-scale-" + str(scale) + fileExt,resized)


def main():
    if(len(sys.argv) == 3):
        path = Path(sys.argv[1])
        file = sys.argv[1]
        image = cv.imread(file)
        resize(image, path.stem, path.suffix, int(sys.argv[2]))
    else:
        print("Please run the program with 'python resize.py file scale'")

if(__name__ == '__main__'):
    main()