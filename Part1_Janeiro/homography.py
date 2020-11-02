import cv2
import numpy as np
from numpy.linalg import inv
from random import randrange as rdr

def get_key_points(img, template):
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img, mask = None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(template, mask = None)
    return (keypoints_1, descriptors_1, keypoints_2, descriptors_2)

def get_matched_key_points(keypoints_1, descriptors_1, keypoints_2, descriptors_2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)

    xy1 = []
    xy2 = []
    for k in matches:
        xy1.append(keypoints_1[k.queryIdx].pt)
        xy2.append(keypoints_2[k.trainIdx].pt)

    return [xy1, xy2]


def get_homography(frame, template):
    keypoints_1, descriptors_1, keypoints_2, descriptors_2 = get_key_points(frame, template)
    points = get_matched_key_points(keypoints_1, descriptors_1, keypoints_2, descriptors_2)
    return ransac(points=points, dim=template.shape[0:2], number_of_inliers_threshold=0.50, distance_between_inliers_and_original_points_threshold = 2) 



def ransac(points, dim, number_of_inliers_threshold , distance_between_inliers_and_original_points_threshold):
    max_inliers = 0
    best_H = None
    video = 0
    for i in range(1000):
        number_of_inliers = 0
        
        H = calculate_homography_matrix(points)

        try:
            number_of_inliers = get_number_of_inliers(H, points, dim, distance_between_inliers_and_original_points_threshold)
        except:
            pass # If A is singular we can't get H


        max_inliers, best_H = (number_of_inliers, H) if number_of_inliers > max_inliers else (max_inliers, best_H)

        if max_inliers >= number_of_inliers_threshold*len(points[video]): # if we get more than the threshold percentage for inliers
            return best_H
    else: # H was not found 
        percentage_of_inliers = max_inliers/(len(points[video]))
        print('\nFound a matrix H with ' + str(100*percentage_of_inliers) + '\% of inliers')
        return [percentage_of_inliers, best_H]



def euclidean_distance(point_1, point_2):
    return np.hypot(point_1[0] - point_2[0], point_1[1] - point_2[1])


def get_number_of_inliers(H, xy, dim, distance_between_inliers_and_original_points_threshold):
    
    x = 0   # x coordinate
    y = 1   # y coordinate
    number_of_inliers = 0

    for i in range(len(xy[0])):
        
        video_point = xy[0][i]
        template_point = xy[1][i]

        video_point_projected = np.array([video_point[x], video_point[y], 1])

        template_point_projected = H@video_point_projected

        # normalize
        u, v = template_point_projected[0:2]/template_point_projected[2]

        if ((u < 0) or (u >= dim[1]) or (v < 0) or (v >= dim[0])):  # outside of the template paper
            continue

        template_point_projected = np.array([ u, v ])
 
        
        distance_between_projected_and_real_point = euclidean_distance(template_point_projected, template_point)

        
        if distance_between_projected_and_real_point < distance_between_inliers_and_original_points_threshold:    # if distance to the real point is under threshold
            number_of_inliers += 1
        
    return number_of_inliers



def calculate_homography_matrix(points):
    x = 0 # x coordinate
    y = 1 # y coordinate
    video_frame_index = 0 # The frame is the first in the points array (composed of two lists, one for the frame and the other for the template)
    template_index = 1 # The template is the second in the points array
    A = []
    B = []
    for point_index in range(4):
        random_point = rdr(0, len(points[video_frame_index]))
        point = (points[video_frame_index][random_point], points[template_index][random_point])
        video_frame_x = point[video_frame_index][x]
        video_frame_y = point[video_frame_index][y]
        template_x = point[template_index][x]
        template_y = point[template_index][y]

        # Build A matrix
        A.append([video_frame_x, video_frame_y, 1, 0, 0, 0, -video_frame_x*template_x, -video_frame_y*template_x])
        A.append([0, 0, 0, video_frame_x, video_frame_y, 1, -video_frame_x*template_y, -video_frame_y*template_y])

        # Build B matrix
        B.append(template_x)
        B.append(template_y)

    A = np.array(A)
    B = np.array(B)
    H = None
    
    try: # if it's a singular matrix it will give an error
        H = inv(A.T@A)@(A.T@B)
        H = np.append(H,1)  # we use h9=1
        H = H.reshape(3,3)
    except:
        pass

    return H


 