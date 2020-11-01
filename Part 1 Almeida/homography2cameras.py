import cv2
import numpy as np
from numpy.linalg import inv
from random import randrange as rdr

'''
Manage to find homography that has at least {tresh_num_inliers} inliers.
It is classified as inlier when its SSE is under than {tresh_dist_inliers} pixels.
'''
def getHomography(frame: np.array, template: np.array, tresh_num_inliers=0.50, tresh_dist_inliers=5,
                max_iterations=1000, min_inliers=10, debug=False)  -> ( np.array ):

    points = computeMatchPoints( frame, template)
    homography = ransac( points ,template.shape[0:2], tresh_num_inliers,
                         tresh_dist_inliers,max_iterations, min_inliers, debug) 
    return homography

'''
Run SIFT algoritm
'''
def findFeatures( frame: np.array ) -> ( tuple ):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute( image = frame, mask = None )
    return (keypoints, descriptors )

'''
Compute macth points
'''
def computeMatchPoints( frame: np.array, template: np.array ) -> ( list ):

    keypoints_1, descriptors_1 = findFeatures( frame )
    keypoints_2, descriptors_2 = findFeatures( template )

    # Feature matching
    bf = cv2.BFMatcher( normType = cv2.NORM_L2, # Euclidean norm
                        crossCheck = True) # Returns only those matches with value (i,j) such that i-th descriptor in set A has j-th descriptor in set B as the best match and vice-versa
    matches = bf.match( descriptors_1, descriptors_2)

    # Append the match points between both images
    points = [[],[]]
    for k in matches:
        points[0].append(keypoints_1[k.queryIdx].pt)  # queryIdx - Index of the descriptor in query descriptors
        points[1].append(keypoints_2[k.trainIdx].pt)  # trainIdx - Index of the descriptor in train descriptors

    return points

'''
Compute Homography
'''
def computeHomography( xy: list ) -> ( np.array ):

    x = 0   # xx coordinate
    y = 1   # yy coordinate
    frame = 0   # video frame
    template = 1    # template frame

    # Compute homography matrix -> A@H = B
    A = []
    B = []

    # Uses 4 points because the homograpgy has 8 degrees of freedom 
    for i in range(4):

        # Get random points
        random_point = rdr( 0, len( xy[frame] ) )
        random_point = (xy[frame][random_point],xy[template][random_point])
        x_frame = random_point[frame][x]
        y_frame = random_point[frame][y]
        x_template = random_point[template][x]
        y_template = random_point[template][y]

        A.append( [ x_frame, y_frame, 1, 0, 0, 0, -x_frame*x_template, -y_frame*x_template ] )
        A.append( [ 0, 0, 0, x_frame, y_frame, 1, -x_frame*y_template, -y_frame*y_template ] )

        # coordinates at template frame 
        B.append( x_template )
        B.append( y_template )

    A = np.array(A)
    B = np.array(B)
    H = None
    
    try: # in case of singular matrix

        # Sum of Least Squares
        H = inv(A.T@A)@(A.T@B)
        H = np.append(H,1)  # appends H9 which was the normalization value
        H = H.reshape(3,3)

    except:
        pass

    return H

'''
Run Ransac Algoritm
'''
def ransac( xy, dim, tresh_num_inliers, tresh_dist_inliers, max_iterations, min_inliers, debug ) -> ( np.array ):

    max_inliers = 0
    H_best = None
    # Run the algoritm {max_iterations} times
    for i in range(max_iterations):
        num_inliers = 0

        # get homography matrix
        H = computeHomography( xy )
        
        try: num_inliers = getNumInliers( H, xy, dim, tresh_num_inliers ,tresh_dist_inliers )
        except: pass    # Sometimes A is a singluar matrix and H can not be found

        # Need at least {min_inliers} points
        if num_inliers < min_inliers:
            continue

        max_inliers, H_best = ( num_inliers, H)  if num_inliers > max_inliers else (max_inliers, H_best)

        if max_inliers >= tresh_num_inliers*len(xy[0]): # if we get at least tresh_dist_inliers% of inlier points
            if debug: print(f'\nAn Homography which fulfil all the requirements was found :)')
            return H_best

    else:   # can´t find H
        percentage = max_inliers/len(xy[0])
        if debug: print(f'\nFound a matrix H with {100*percentage:.1f}% inliers')
        return [percentage, H_best]
        
'''
Computes the number of inliers
'''
def getNumInliers( H, xy, dim, tresh_num_inliers ,tresh_dist_inliers ) -> ( int ):
    
    x = 0   # xx coordinate
    y = 1   # yy coordinate
    num_inliers = 0

    distance_points = lambda x1, x2: np.hypot( x1[0] - x2[0], x1[1] - x2[1] )

    for i in range(len(xy[0])):
        
        point_video = xy[0][i]
        point_template = xy[1][i]

        point_video_projective = np.array( [ point_video[x], point_video[y], 1 ] )

        point_template_projective = H@point_video_projective

        # Normalize
        u, v = point_template_projective[0:2]/point_template_projective[2]

        # Outside the template
        if ( (u < 0) or (u >= dim[1]) or (v < 0) or (v >= dim[0]) ):
            breakpoint
            continue

        point_template_projective = np.array( [ u, v ] )
 
        # Geometric distance
        sse = distance_points(point_template_projective, point_template)
        # In/Out lier, decision
        if sse < tresh_dist_inliers:
            num_inliers += 1
        
    return num_inliers

'''
Generate a discret sequence
'''
def infinite_sequence() -> (int):
    num = 0
    while True:
        yield num
        num += 1