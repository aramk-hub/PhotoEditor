import numpy as np
import cv2 as cv


img = cv.imread('./Denoise/DSCF0413.jpg')
dst = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)
cv.imwrite('denoised.png', dst)
