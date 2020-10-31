import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from random import randrange as rdr
from filter import *
import pickle as pk


def getHomography(img, template):

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

    xy = [xy1, xy2]
    return ransac( xy=xy, dim=template.shape[0:2], tresh_num_inliers=0.50, tresh_dist_inliers = 10) 


def ransac(xy,dim,tresh_num_inliers,tresh_dist_inliers):

    max_inliers = 0
    best_H = 0

    video = 0   # video frame
    template = 1    # template frame
    x = 0   # xx coordinate
    y = 1   # yy coordinate
    for i in range(1000):

        # compute homography matrix -> A@H = B
        A = []
        B = []

        for elem in range(4):

            random_point = rdr( 0, len( xy[video] ) )

            point = (xy[video][random_point],xy[template][random_point])

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

        if max_inliers >= tresh_num_inliers*len(xy[video]): # if we get at least tresh_dist_inliers% of inlier points
            print(f'Nice H :)')
            return H

    else:   # can´t find H
        print('cant find H')
        return None
        

def getNumInliers( H, xy, dim, tresh_num_inliers ,tresh_dist_inliers, max_inliers ):
    
    x = 0   # xx coordinate
    y = 1   # yy coordinate
    inliers = 0

    distance_points = lambda x1, x2: np.hypot( x1[0] - x2[0], x1[1] - x2[1] )

    for i in range(len(xy[0])):
        
        point_video = xy[0][i]
        point_template = xy[1][i]

        point_video_projective = np.array( [ point_video[x], point_video[y], 1 ] )

        point_template_projective = H@point_video_projective

        # normalize
        u, v = point_template_projective[0:2]/point_template_projective[2]

        if ( (u < 0) or (u >= dim[1]) or (v < 0) or (v >= dim[0]) ):  # outside the template
            breakpoint
            continue

        point_template_projective = np.array( [ u, v ] )
 
        
        sse = distance_points(point_template_projective, point_template)

        
        if sse < tresh_dist_inliers:    # if distance to the real point is under than 5px
            inliers += 1
        
    return inliers

