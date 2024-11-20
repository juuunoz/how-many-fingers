import numpy as np
import math
import cv2 as cv

cap = cv.VideoCapture(0)

win_name_raw = 'raw camera'
win_name = 'camera'
cv.namedWindow(win_name_raw, cv.WINDOW_NORMAL)
cv.namedWindow(win_name, cv.WINDOW_NORMAL)

# Idea: Use AI to locate square adding more dynamic hand recognition
# Idea: Use AI to personalize which square it recommends

while cv.waitKey(1) != 27:
    # Read in a frame from the camera
    ret, img = cap.read()
    
    cv.rectangle(img, (450, 550), (100, 100), (0, 0, 255), 0)
    if not ret: 
        break
    
    # Threshold the region of interest to bring out the hand silhouette
    img_cropped = img[100:550, 100:450]
    img_grey = cv.cvtColor(img_cropped, cv.COLOR_BGR2GRAY)
    img_blurred = cv.GaussianBlur(img_grey, (35, 35), 0)
    _, img_thresholded = cv.threshold(img_blurred, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Contouring
    contours, hierarchy = cv.findContours(img_thresholded, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img_thresholded = cv.cvtColor(img_thresholded, cv.COLOR_GRAY2BGR)
    cv.drawContours(img_thresholded, contours, -1, (0, 255, 0), 3)

    # Processing contours
    max_contour = max(contours, key=lambda x: cv.contourArea(x))
    x, y, w, h = cv.boundingRect(max_contour)
    cv.rectangle(img_cropped, (x, y), (x + w, y + h), (0, 0, 255), 1)

    cv.imshow(win_name_raw, img)
    cv.imshow(win_name, img_thresholded)

cap.release()
cv.destroyWindow(win_name_raw)
cv.destroyWindow(win_name)
