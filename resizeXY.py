import cv2 as cv
import numpy as np
import sys
from pathlib import Path

def resize(image, fileName, fileExt, xScale, yScale):
        resized = cv.resize(image, None, fx=xScale/100, fy=yScale/100, interpolation=cv.INTER_CUBIC)
        cv.imwrite(fileName + "-xScale-" + str(xScale) + "%-yScale-" + str(yScale) + "%" + fileExt,resized)


def main():
    if(len(sys.argv) == 4):
        path = Path(sys.argv[1])
        file = sys.argv[1]
        image = cv.imread(file)
        resize(image, path.stem, path.suffix, int(sys.argv[2]), int(sys.argv[3]))
    else:
        print("Please run the program with 'python resize.py file xScale yScale'")

if(__name__ == '__main__'):
    main()