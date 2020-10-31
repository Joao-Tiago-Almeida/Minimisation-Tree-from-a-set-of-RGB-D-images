import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.datasets import make_regression

def reorder(my_points):
     
    my_points = my_points.reshape((4, 2))
    my_points_ordered = np.zeros((4, 1, 2), dtype=np.int32)
    add = my_points.sum(1)
 
    my_points_ordered[0] = my_points[np.argmin(add)]
    my_points_ordered[3] = my_points[np.argmax(add)]
    diff = np.diff(my_points, axis=1)
    my_points_ordered[1] = my_points[np.argmin(diff)]
    my_points_ordered[2] = my_points[np.argmax(diff)]
 
    return my_points_ordered

def make_mask(biggest, shape):

    biggest = reorder(biggest)

    x1 = min(biggest[0][0][0], biggest[2][0][0])
    x2 = max(biggest[1][0][0], biggest[3][0][0])
    y1 = min(biggest[0][0][1], biggest[1][0][1])
    y2 = max(biggest[2][0][1], biggest[3][0][1])

    mask = np.zeros([shape[0], shape[1]])

    for x in range(x1,x2+1):
        for y in range(y1,y2+1):
            mask[x][y] = 1

    print(mask[400][400])

    return mask
    
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


#reading image
pathfile = "images/resized_paperview.jpg"
template_file = "images/resized_template.png"

img = cv2.imread(pathfile)
template = cv2.imread(template_file)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
'''
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
img_threshold = cv2.Canny(img_blur, 100, 200)

template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template_blur = cv2.GaussianBlur(template_gray, (5, 5), 1)
template = cv2.Canny(template_blur, 100, 200)

#cv2.imshow('image', img)
#cv2.imshow('template', template)

img_contours = img.copy()
img_biggest = img.copy()

img_dilated = cv2.dilate(img_threshold, np.ones((5,5)), iterations=2)
img_threshold = cv2.erode(img_dilated, np.ones((5,5)), iterations=1)

img_dilated = cv2.dilate(template, np.ones((5,5)), iterations=2)
template = cv2.erode(img_dilated, np.ones((5,5)), iterations=1)

contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 10)

cv2.imshow('contours', img_contours)

biggest, _ = get_biggest_contour(contours)
cv2.drawContours(img_biggest, biggest, -1, (0, 255, 0), 10)

cv2.imshow('biggest', img_biggest)

# print(make_mask(biggest, img_biggest.shape))

img = img_threshold

cv2.imshow('imagem1', img_threshold)

cv2.waitKey(0)
cv2.destroyAllWindows()

exit()
'''

#cv2.imshow('image', img)
#cv2.imshow('template', template)

#sift
sift = cv2.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(template,None)

x = []
y = []
for k in keypoints_1:
    x.append(k.pt[0])
    y.append(k.pt[1])

#print(keypoints_1[0])

x_inv = []
y_inv = []
for k in keypoints_2:
    x_inv.append(k.pt[0])
    y_inv.append(k.pt[1])

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

#matches_inv = bf.match(descriptors_2, descriptors_1)
#matches_inv = sorted(matches_inv, key = lambda x:x.distance)

#print(len(matches))

img3 = cv2.drawMatches(img, keypoints_1, template, keypoints_2, matches[:200], template, flags=2)
plt.imshow(img3)

# cv2.imshow('final', img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
img_inv = cv2.drawMatches(template, keypoints_2, img, keypoints_1, matches_inv[:200], img, flags=2)
cv2.imshow('final_inv', img_inv)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

x_final = []
y_final = []

'''
print(matches[:200])
for i in matches[:200]:
    #print(i.trainIdx)
    #x_final.append(keypoints_1[i.trainIdx].pt[0])
    #y_final.append(keypoints_1[i.trainIdx].pt[1])
    x_final.append(keypoints_1[i.trainIdx])
    y_final.append(keypoints_2[i.trainIdx])'''

'''for i in range(200):
    if(keypoints_1[matches[])'''


for k in matches[:200]:
    x_final.append(keypoints_1[k.queryIdx].pt)
    y_final.append(keypoints_2[k.trainIdx].pt)
    

x_final = np.array(x_final).reshape(-1,2)
y_final = np.array(y_final).reshape(-1,2)


reg = RANSACRegressor(random_state=0).fit(x_final, y_final)
print(reg.score(x_final, y_final))

inlier_mask = reg.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

print(len(inlier_mask))
print(len(x_final))



lw = 2
plt.scatter(x_final[inlier_mask][:,0], x_final[inlier_mask][:,1], color='yellowgreen', marker='.',
            label='Inliers')
plt.scatter(x_final[outlier_mask][:,0], x_final[outlier_mask][:,1], color='gold', marker='.',
            label='Outliers')

plt.xlabel("Input")
plt.ylabel("Response")
plt.show()
