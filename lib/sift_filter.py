import numpy as np
import cv2 as cv

img = cv.imread('./../data/home.jpg')
cv.imshow('Original Image', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp = sift.detect(gray, None)

img = cv.drawKeypoints(gray, kp, img)

cv.imshow('SIFT Filter', img)

# sift = cv.SIFT_create()
# kp, des = sift.detectAndCompute(gray, None)

cv.waitKey(0)
print(cv.__version__)
