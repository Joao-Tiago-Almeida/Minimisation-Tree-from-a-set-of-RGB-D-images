import cv2
import numpy as np
from numpy.linalg import inv
from random import randrange as rdr

'''
Manage to find homography that has at least {thresh_num_inliers} inliers.
It is classified as inlier when its SSE is under than {thresh_dist_inliers} pixels.
'''
def getHomography(frame: np.array, template: np.array, thresh_num_inliers=0.50, thresh_dist_inliers=5,
                max_iterations=1000, min_inliers=10, debug=False)  -> ( np.array ):

    points = computeMatchPoints( frame, template)
    homography = ransac( points ,template.shape[0:2], thresh_num_inliers,
                         thresh_dist_inliers,max_iterations, min_inliers, debug) 

    # Fits the Homography to the inliers
    H = homography[1]
    new_homography = computeHomography( getNumInliers( H, points, template.shape[0:2],thresh_dist_inliers, get_num_inliers=False )
                                                       , is_random=False )
    
    # Computes the new percentage
    inliers =  getNumInliers( new_homography, points, template.shape[0:2],thresh_dist_inliers, get_num_inliers=True )

    if(inliers >= homography[2]):    # the new matrix has more inliers
            imp = inliers - homography[2]
            H = new_homography

    else:
        inliers = homography[2]
        H = homography[1]
    
    percentage = 100*inliers/len(points[0])

    if(homography[0] or percentage >= 100*thresh_num_inliers):  # found a nice H matrix

        if debug:
            print(f'\nAn Homography which fulfils all the requirements was found :) with {homography[2]} inliers.')
            if imp >= 0: print(f'The matrix was fitted getting {imp} more point{"" if imp == 1 else "s"}, resulting on a total of {inliers} inliers.')
            print(f'The final percentage of inliers is {percentage:.1f}% !!')

        return H
    else:
        if debug: print(f"\nFound a matrix H with {percentage:.1f}% :( I'll keep trying")

        return [inliers, H]


'''
Run SIFT algorithm
'''
def findFeatures( frame: np.array ) -> ( tuple ):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute( image = frame, mask = None )
    return (keypoints, descriptors )

'''
Compute match points
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
def computeHomography( xy: list, is_random: bool ) -> ( np.array ):

    x = 0   # xx coordinate
    y = 1   # yy coordinate
    frame = 0   # video frame
    template = 1    # template frame

    # Compute homography matrix -> A@H = B
    A = []
    B = []

    # Uses 4 points because the homography has 8 degrees of freedom 
    loop = 4 if is_random else len(xy[0])
    for i in range(loop):

        if is_random:   # Get random points
            random_point = rdr( 0, len( xy[frame] ) )
            random_point = (xy[frame][random_point],xy[template][random_point])
        else:   # iterates for all points
            random_point = [xy[0][i], xy[1][i]]

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
Run Ransac Algorithm
'''
def ransac( xy, dim, thresh_num_inliers, thresh_dist_inliers, max_iterations, min_inliers, debug ) -> ( np.array ):

    max_inliers = 0
    H_best = None
    # Run the algorithm {max_iterations} times
    for i in range(max_iterations):
        num_inliers = 0

        # get homography matrix
        H = computeHomography( xy, is_random=True )
        
        try: num_inliers = getNumInliers( H, xy, dim, thresh_dist_inliers, get_num_inliers=True )
        except: pass    # Sometimes A is a singular matrix and H can not be found

        # Need at least {min_inliers} points
        if num_inliers < min_inliers:
            continue

        max_inliers, H_best = ( num_inliers, H)  if num_inliers > max_inliers else (max_inliers, H_best)

        if max_inliers >= thresh_num_inliers*len(xy[0]): # if we get at least thresh_dist_inliers% of inlier points
            
            return [True, H_best, max_inliers]

    else:   # can´t find H
        return [False, H_best, max_inliers]
        
'''
Computes the number of inliers
'''
def getNumInliers( H, xy, dim ,thresh_dist_inliers, get_num_inliers ) -> ( int ):
    
    x = 0   # xx coordinate
    y = 1   # yy coordinate
    num_inliers = 0
    inliers = [[],[]]

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
            continue

        point_template_projective = np.array( [ u, v ] )
 
        # Geometric distance
        sse = distance_points(point_template_projective, point_template)
        # In/Out lier, decision
        if sse < thresh_dist_inliers:
            num_inliers += 1
            if not get_num_inliers:
                inliers[0].append(point_video)
                inliers[1].append(point_template)
        
    return num_inliers if get_num_inliers else inliers

'''
Generate a discret sequence
'''
def infinite_sequence() -> (int):
    num = 0
    while True:
        yield num
        num += 1