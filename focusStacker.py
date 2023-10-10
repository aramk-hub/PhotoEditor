import cv2 as cv
import numpy as np

def alignImages(images):

    alignedImages = list()
    detector = cv.xfeatures2d.SIFT_create()

    # Appened reference image, detect and compute descriptors
    alignedImages.append(images[0])
    ref_grey = cv.cvtColor(images[0], cv.COLOR_BGR2GRAY)
    ref_kp, ref_desc = detector.detectAndCompute(ref_grey, None)

    # Go through the other images, aligning each one using feature matching
    for i in range(1, len(images)):
        
        kpi, kpi_desc = detector.detectAndCompute(images[i], None)

        bf = cv.BFMatcher()
        # Match & ratio test
        matches = cv.knnMatch(kpi_desc, ref_desc, k=2)
        good_images = list()
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                good_images.append(m)

        matches = good_images

        # Find Homography
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = ref_kp[match.queryIdx].pt
            points2[i, :] = kpi[match.trainIdx].pt
        
        hom, mask = cv.findHomography(points1, points2, cv.RANSAC)

        result = cv.warpPerspective(images[i], hom, (images[i].shape[1], images[i].shape[0]), flags=cv.INTER_LINEAR)

        alignedImages.append(result)

        cv.imwrite("aligned{}.png".format(i), result)

    return alignedImages


