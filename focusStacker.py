import cv2 as cv
import numpy as np
import os

def alignImages(images):

    alignedImages = list()
    detector = cv.SIFT_create()

    # Appened reference image, detect and compute descriptors
    alignedImages.append(images[0])
    ref_grey = cv.cvtColor(images[0], cv.COLOR_BGR2GRAY)
    ref_kp, ref_desc = detector.detectAndCompute(ref_grey, None)

    # Go through the other images, aligning each one using feature matching
    for i in range(1, len(images)):
        
        kpi, kpi_desc = detector.detectAndCompute(images[i], None)

        bf = cv.BFMatcher()
        # Match & ratio test
        matches = bf.knnMatch(kpi_desc, ref_desc, k=2)
        good_images = list()
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                good_images.append(m)

        matches = good_images

        # Find Homography
        points1 = np.zeros((len(matches), 1, 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 1, 2), dtype=np.float32)
        

        for j in range(len(matches)):
            points2[j] = ref_kp[matches[j].trainIdx].pt
            points1[j] = kpi[matches[j].queryIdx].pt
        
        hom, mask = cv.findHomography(points1, points2, cv.RANSAC, ransacReprojThreshold=2.0)

        result = cv.warpPerspective(images[i], hom, (images[i].shape[1], images[i].shape[0]), flags=cv.INTER_LINEAR)

        alignedImages.append(result)

        cv.imwrite("aligned{}.png".format(i), result)

    return alignedImages

# Find Laplacian gradient to find the in-focus parts of the image
def laplace(image):
    bs = 5
    ks = 5
    return cv.Laplacian(cv.GaussianBlur(image, (bs, bs), 0), cv.CV_64F, ksize=ks)


def stack_focuses(images):
    images = alignImages(images)

    laplacians = list()

    for image in images:
        greyed = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        laplacians.append(laplace(greyed))

    laplacian = np.asarray(laplacians)
    out = np.zeros(shape=images[0].shape, dtype=images[0].dtype)
    abs_laps = np.absolute(laplacians)

    # # Iterate through every pixel, setting the output pixel to the px with largest gradient
    for y in range(images[0].shape[0]):
        for x in range(images[0].shape[1]):
            gradients = [image[y,x] for image in abs_laps]
            out_color = images[gradients.index(max(gradients))][y,x]
            out[y,x] = out_color


    # cv.imshow('out', output)
    cv.imwrite('./Outputs/focused.png', out)

if __name__ == "__main__":
    images = list()
    for image in os.listdir("./Focusing"):
        images.append(cv.imread("./Focusing/{}".format(image)))
    
    stack_focuses(images)
        