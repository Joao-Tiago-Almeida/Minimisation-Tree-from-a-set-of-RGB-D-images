# Helper methods for PIV project Part 1.

import cv2
import numpy as np


def is_rectangle(approx):
  if len(approx) == 4: # If it has 4 sides it's a rectangle
    return True
  else:
    return False

def get_biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000: # Minimum area to eliminate unwanted noise
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True) # Get an approximate polygon from contour shape
            if area > max_area and is_rectangle(approx):
                biggest = approx
                max_area = area
    return biggest,max_area

# Make sure the contours points are in order, i.e: top left is 0, top right is 1, bottom left is 2 and bottom right is 3. 
# The detection can have a different order and the rendered image would have a different orientation
def reorder(my_points):
 
    my_points = my_points.reshape((4, 2))
    my_points_ordered = np.zeros((4, 1, 2), dtype=np.int32)
    add = my_points.sum(1)
 
    my_points_ordered[0] = my_points[np.argmin(add)]
    my_points_ordered[3] = my_points[np.argmax(add)]
    diff = np.diff(my_points, axis=1)
    my_points_ordered[1] = my_points[np.argmin(diff)]
    my_points_ordered[2] = my_points[np.argmax(diff)]


    # diff = np.diff(my_points, axis=1)
    # my_points_ordered[0] = my_points[np.argmax(diff)]
    # my_points_ordered[3] = my_points[np.argmin(diff)]
    
    # my_points_ordered[1] = my_points[np.argmin(add)]
    # my_points_ordered[2] = my_points[np.argmax(add)]
 
    return my_points_ordered

## TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    # if len(lables) != 0:
    #     eachImgWidth= int(ver.shape[1] / cols)
    #     eachImgHeight = int(ver.shape[0] / rows)
    #     print(eachImgHeight)
    #     for d in range(0, rows):
    #         for c in range (0,cols):
    #             cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
    #             cv2.putText(ver,lables[d],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver