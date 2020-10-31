import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv


#reading image
pathfile = "images/resized_paperview.jpg"
template_file = "images/resized_template.png"

img = cv2.imread(pathfile)
template = cv2.imread(template_file)


#cv2.imshow('image', img)
#cv2.imshow('template', template)

#sift
sift = cv2.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(template,None)

getPoints = lambda l: [elem.pt for elem in l]

xy1 = getPoints(keypoints_1)
xy2 = getPoints(keypoints_2)

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)

xy1 = []
xy2 = []
for k in matches:
    xy1.append(keypoints_1[k.queryIdx].pt)
    xy2.append(keypoints_2[k.trainIdx].pt)
xy = zip(xy1,xy2)

#  sort xy by the distance to center
center = (img.shape[1]/2, img.shape[0]/2)
distance_points = lambda a: np.hypot( center[0] - a[0][0], center[1] - a[0][1] )
xy = sorted(xy, key=distance_points)

xy1 = []
xy2 = []
for i in xy:
    xy1.append(i[0])
    xy2.append(i[1])

xy = [xy1, xy2]
breakpoint

def ransac(xy,dim,tresh_num_inliers,tresh_dist_inliers):

    max_inliers = 0

    video = 0   # video frame
    template = 1    # template frame
    x = 0   # xx coordinate
    y = 1   # yy coordinate
    for i in range(len(xy[0])-3):

        XY = ( xy[video][i:i+4], xy[template][i:i+4] ) # get four points

        # compute homography matrix -> A@H = B
        A = []
        B = []

        
        for elem in range(4):

            point = (XY[video][elem],XY[template][elem])

            A.append( [ point[video][x], point[video][y], 1, 0, 0, 0, -point[video][x]*point[template][x], -point[video][y]*point[template][x] ] )
            A.append( [ 0, 0, 0, point[video][x], point[video][y], 1, -point[video][x]*point[template][y], -point[video][y]*point[template][y] ] )
            
            # coordinates at template frame 
            B.append( point[template][x] )
            B.append( point[template][y] )

        A = np.array(A)
        B = np.array(B)
        
        try: #in case of singular matrix

            # Sum of Least Squares
            H = inv(A.T@A)@(A.T@B)
            H = np.append(H,1)
            H = H.reshape(3,3)

        except:
            continue
        
        ninliers = getNumInliers( H, xy, dim, tresh_num_inliers ,tresh_dist_inliers, max_inliers )

        max_inliers = ninliers if ninliers > max_inliers else max_inliers

        if max_inliers > tresh_num_inliers*len(xy[video]): # if we get at least tresh_dist_inliers% of inlier points
            return H

    else:   # can´t find H
        pass
        
        
# the name is already speacified ...
def getNumInliers( H, xy, dim, tresh_num_inliers ,tresh_dist_inliers, max_inliers ):
    
    x = 0   # xx coordinate
    y = 1   # yy coordinate
    inliers = 0
    for i in range(len(xy[0])):
        
        point_video = xy[0][i]
        point_template = xy[1][i]

        point_video_projective = np.array( [ point_video[x], point_video[y], 1 ] )

        point_template_projective = H@point_video_projective

        # normalize
        u, v = point_template_projective[0:2]/point_template_projective[2]

        if u < 0 : u = 0
        elif u > dim[1]: u = dim[1]
        
        if v < 0 : v = 0
        elif v > dim[0]: v = dim[0]

        point_template_projective = np.array( [ u, v ] ) # it is treated (y, x)
 
        distance_points = lambda x1, x2: np.hypot( x1[0] - x2[0], x1[1] - x2[1] )
        sse = distance_points(point_template_projective, point_template)

        
        if sse < tresh_dist_inliers:    # if distance to the real point is under than 5px
            inliers += 1
        
    return inliers


final_H = ransac( xy=xy, dim=template.shape[0:2], tresh_num_inliers=0.50, tresh_dist_inliers = 10) 

breakpoint

new_template = np.empty(template.shape, dtype=np.uint8)

for y in range(img.shape[0]):
    for x in range(img.shape[1]):

        XY_video = np.array( [x, y, 1] )

        XY_template = final_H@XY_video
        u ,v = XY_template[0:2]/XY_template[2]

        if 0 < u and u < new_template.shape[1] and 0 < v and v < new_template.shape[0]:
            new_template[int(v)][int(u)] = img[y][x]


cv2.imshow('image', new_template)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
link with:
theorical support -> https://drive.google.com/file/d/1ntGJz4xYIeyuvs4ztMzfNbs91LpQ8Qs0/view?usp=sharing
practical support -> https://github.com/hughesj919/HomographyEstimation/blob/master/Homography.py
'''