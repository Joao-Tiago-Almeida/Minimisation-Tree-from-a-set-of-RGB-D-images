import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.datasets import make_regression

#reading image
pathfile = "images/paper_view3.png"
template_file = "images/template2.png"

img = cv2.imread(pathfile)
template = cv2.imread(template_file)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
'''
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
img = cv2.Canny(img_blur, 100, 200)

template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template_blur = cv2.GaussianBlur(template_gray, (5, 5), 1)
template = cv2.Canny(template_blur, 100, 200)
'''
'''
cv2.imshow('image', img)
cv2.imshow('template', template)
'''

#sift
sift = cv2.xfeatures2d.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(template,None)

x = []
y = []
for k in keypoints_1:
    x.append(k.pt[0])
    y.append(k.pt[1])

print(keypoints_1[0])

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

print(len(matches))

img3 = cv2.drawMatches(img, keypoints_1, template, keypoints_2, matches[:200], template, flags=2)
plt.imshow(img3)
'''
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
'''

x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)

reg = RANSACRegressor(random_state=0).fit(x, y)
print(reg.score(x, y))

inlier_mask = reg.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

print(len(inlier_mask))
print(len(y))

lw = 2
plt.scatter(x[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
            label='Inliers')
plt.scatter(x[outlier_mask], y[outlier_mask], color='gold', marker='.',
            label='Outliers')

plt.xlabel("Input")
plt.ylabel("Response")
plt.show()