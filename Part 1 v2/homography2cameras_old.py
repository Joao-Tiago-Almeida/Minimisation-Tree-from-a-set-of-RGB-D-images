import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from random import randrange as rdr
from filter import *
import pickle as pk


#reading image

pathfile = "images/foto.png"
template_file = "images/template2.png"

img =  pk.load(open('frameN1.p', 'rb')) #cv2.imread(pathfile)
template = cv2.imread(template_file)

resize = lambda x: cv2.resize(x, (int(x.shape[1]/2) , int(x.shape[0]/2) ))

template = resize(template)
img = resize(img)


'''
cv2.imshow('image', img)
cv2.imshow('template', template)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

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


#  sort xy by the distance to center
# xy = zip(xy1,xy2)
# center = (img.shape[1]/2, img.shape[0]/2)
# distance_points = lambda a: np.hypot( center[0] - a[0][0], center[1] - a[0][1] )
# xy = sorted(xy, key=distance_points)

# xy1 = []
# xy2 = []
# for i in xy:
#     xy1.append(i[0])
#     xy2.append(i[1])

xy = [xy1, xy2]
breakpoint

def ransac(xy,dim,tresh_num_inliers,tresh_dist_inliers):

    max_inliers = 0
    best_H = 0

    video = 0   # video frame
    template = 1    # template frame
    x = 0   # xx coordinate
    y = 1   # yy coordinate
    for i in range(10000):

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

        max_inliers, best_H = ( ninliers, H)  if ninliers > max_inliers else (max_inliers, best_H)

        if max_inliers >= tresh_num_inliers*len(xy[video]): # if we get at least tresh_dist_inliers% of inlier points
            print(f'Nice H :)')
            return H

    else:   # can´t find H
        print(f'found a matrix H with {100*max_inliers/len(xy[0]):.1f}% inliers')
        return best_H
        
        
# the name is already speacified ...
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



final_H = ransac( xy=xy, dim=template.shape[0:2], tresh_num_inliers=0.75, tresh_dist_inliers = 5) 

pk.dump(final_H, open('HN2.p', 'wb'))

breakpoint

#final_H = pk.load(open('Hn1.p', 'rb'))
img_warp_colored = cv2.warpPerspective(img, final_H, (int(1*template.shape[1]), int(1*template.shape[0])))
cv2.imshow('filtered_image', img_warp_colored)
cv2.waitKey(0)

pk.dump(img_warp_colored, open('frameN2.p', 'wb'))

# img1 = median_2D(img_warp_colored)
# cv2.imshow('img1', img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

H1 = pk.load(open('Hn1.p', 'rb'))
final_H = final_H@inv(H1)
pk.dump(final_H, open('HT.p', 'wb'))


'''

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


pk.dump(new_template, open('frame.p', 'wb'))

new_template = cv2.cvtColor(new_template, cv2.COLOR_BGR2GRAY)
filtered_image = median_2D(new_template)

cv2.imshow('filtered_image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
link with:
theorical support -> https://drive.google.com/file/d/1ntGJz4xYIeyuvs4ztMzfNbs91LpQ8Qs0/view?usp=sharing
practical support -> https://github.com/hughesj919/HomographyEstimation/blob/master/Homography.py
'''