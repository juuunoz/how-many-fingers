import numpy as np
import math
import cv2 as cv

cap = cv.VideoCapture(0)

win_name = 'camera'
cv.namedWindow(win_name, cv.WINDOW_NORMAL)

while cv.waitKey(1) != 27:
    # Read in a frame from the camera
    ret, img = cap.read()
    
    cv.rectangle(img, (450, 550), (100, 100), (0, 0, 255), 0)
    if not ret: 
        break
    
    # Threshold the region of interest to bring out the hand silhouette
    img_cropped = img[100:450, 100:550]
    img_grey = cv.cvtColor(img_cropped, cv.COLOR_BGR2GRAY)
    img_blurred = cv.GaussianBlur(img_grey, (35, 35), 0)
    _, img_thresholded = cv.threshold(img_blurred, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    cv.imshow(win_name, img_thresholded)


cap.release()
cv.destroyWindow(img_thresholded)
