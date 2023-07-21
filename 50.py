# import the necessary packages
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


# load the reference OCR-A image from disk, convert it to grayscale,
# and threshold it, such that the digits appear as *white* on a
# *black* background
# and invert it, such that the digits appear as *white* on a *black*
ref = cv2.imread('data/tq.jpg')
ref = cv2.imread('tmp99.png')
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]




# find contours in the OCR-A image (i.e,. the outlines of the digits)
# sort them from left to right, and initialize a dictionary to map
# digit name to the ROI
refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
refCnts = imutils.grab_contours(refCnts) #适配cv2各个版本.
refCnts = sorted(refCnts, key = cv2.contourArea, reverse = True)[:5]

# refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
digits = {}





# loop over the OCR-A reference contours
for (i, c) in enumerate(refCnts):
	# compute the bounding box for the digit, extract it, and resize
	# it to a fixed size
    (x, y, w, h) = cv2.boundingRect(c)	
    print(1)
    # roi = ref[y:y + h, x:x + w]
	# roi = cv2.resize(roi, (57, 88))

	# # update the digits dictionary, mapping the digit name to the ROI
	# digits[i] = roi