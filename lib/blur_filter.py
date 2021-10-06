import cv2 as cv

img = cv.imread('./../data/lena.jpg')
cv.imshow('Lena original', img)

img_blur = cv.blur(img, (3, 3))
cv.imshow('Lena blurred 3x3', img_blur)

img_gauss = cv.GaussianBlur(img, (7, 7), 31)
cv.imshow('Lena Gaussian-blurred 3x3', img_gauss)

img_med = cv.medianBlur(img, 7)
cv.imshow('Lena Median blurred', img_med)

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_sobel = cv.Sobel(img_gray, cv.CV_8U, 1, 0, 3, 7, 128)
cv.imshow('Lena Sobel', img_sobel)

cv.waitKey(0)
