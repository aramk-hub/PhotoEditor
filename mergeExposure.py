from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse
import os


def loadImExpTime():
    
    path = "./HDR"
    images = list()
    times = np.array([ 20.0, 20.0, 20.0], dtype=np.float32)

    for file in os.listdir(path):
        images.append(cv.imread(os.path.join(path, file)))
    
    return images, times


# First read in images from HDR folder
images, times = loadImExpTime()

# Align input images
alignMTB = cv.createAlignMTB()
alignMTB.process(images, images)

# Estimate camera response with CRF - Camera Response Fucntion
calibrate = cv.createCalibrateDebevec()
response = calibrate.process(images, times)

# Make HDR images
merge_debevec = cv.createMergeDebevec()
hdr = merge_debevec.process(images, times, response)

# Tonemap results
tonemap = cv.createTonemap(2.2)
ldr = tonemap.process(hdr)

# Alternate: Exposure fusion -> Does not need exposure times
merge_mertens = cv.createMergeMertens()
fusion = merge_mertens.process(images)

cv.imwrite('./outputs/fusion.png', fusion * 255)
cv.imwrite('./outputs/ldr.png', ldr * 255)
cv.imwrite('./outputs/hdr.hdr', hdr)
