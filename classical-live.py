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
    # cv.drawContours(img_thresholded, contours, -1, (0, 255, 0), 3)

    # Finding the largest contour, assuming it's the hand
    max_contour = max(contours, key=lambda x: cv.contourArea(x))
    
    # Creating convex hull with countour
    hull = cv.convexHull(max_contour)

    # Drawing contours and hull
    cv.drawContours(img_thresholded, [max_contour], 0, (0, 255, 0), 3)
    cv.drawContours(img_thresholded, [hull], 0, (0, 255, 255), 3)

    defects = cv.convexityDefects(max_contour, cv.convexHull(max_contour, returnPoints=False))

    num_defects = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]

        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        furthest = tuple(max_contour[f][0])

        # Cosine Law to find the angle between two points on the contour and the convexity defect in between
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((furthest[0] - start[0]) ** 2 + (furthest[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - furthest[0]) ** 2 + (end[1] - furthest[1]) ** 2)
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

        if angle <= 90:
            num_defects += 1
            cv.circle(img_thresholded, furthest, 5, [0, 0, 255], -1)

    cv.putText(img, str(num_defects + 1) + "fingers", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
    cv.imshow(win_name_raw, img)
    cv.imshow(win_name, img_thresholded)

cap.release()
cv.destroyWindow(win_name_raw)
cv.destroyWindow(win_name)
