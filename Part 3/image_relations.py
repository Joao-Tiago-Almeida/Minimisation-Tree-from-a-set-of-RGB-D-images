import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange as rdr
from scipy.io import savemat

from scipy.spatial import procrustes


def transformation2cameras( camera: tuple ) -> ( np.array ):
    """
    computes the transformation matrix between 2 cameras [RT]

    return row matrix with 12 values. 
    """

    rgb1 = camera[0]
    rgb2 = camera[1]
    dep1 = camera[2]
    dep2 = camera[3]
    k_rgb = camera[4]
    k_depth = camera[5]
    r_depth2rgb = camera[6]
    t_depth2rgb = camera[7]

    xy = get_matched_key_points( *get_key_points(rgb1, rgb2) )  # returns the keypoints RGB

    pc_dep1 = generate_depth_pc(dep1, k_depth)
    pc_dep2 = generate_depth_pc(dep2, k_depth)

    depth_pc_2_rgb_pc = lambda xyz: np.concatenate( (r_depth2rgb, t_depth2rgb), axis=1)@np.concatenate( (xyz, np.ones((1, xyz.shape[1]), dtype=float) ), axis=0 )

    pc_rgb1 = depth_pc_2_rgb_pc(pc_dep1)
    pc_rgb2 = depth_pc_2_rgb_pc(pc_dep2)

    pc_new = ransac(pc_rgb1, pc_rgb2, xy, rgb1.shape)

    pc_in_matlab( (pc_rgb1, pc_new), ('pc1', 'pc2') )

    rgb_pc_to_rgb_img( pc_new, k_rgb, rgb2 )

    breakpoint

def ransac( pc1: np.array, pc2: np.array, xy: list, dim: tuple ) -> ( tuple ):

    num_itr=100
    percentage_inliers_threshold = 0.5
    dist_inliers_threshold = 0.2

    # best values
    r_best = np.zeros((3,3))
    t_best = np.zeros((3,1))
    sse_best = np.Inf
    pc_try_best = None
    max_inliers = 0


    pos_vect = lambda x, y: round(x)*dim[0] + round(y)
    rigid_transformation = lambda r, t, pc: np.concatenate((r, t), axis=1)@np.concatenate( (pc, np.ones((1, pc.shape[1]), dtype=float)), axis=0 )

    for i in range(num_itr):

        points = 4
        points_vect = [[],[]]

        for i in range(points):

            random_point = rdr( 0, len( xy[0] ) )   # bicates 'aleatori' point

            idx_1 = pos_vect( *xy[0][random_point] )
            idx_2 = pos_vect( *xy[1][random_point] )

            points_vect[0].append(idx_1)
            points_vect[1].append(idx_2)        

        r, t = procrustes_nosso(pc1[:, points_vect[0]], pc2[:, points_vect[1]]) #pc_rgb1, pc_rgb2
        #r,t = batota(pc1[:, points_vect[0]], pc2[:, points_vect[1]]) #pc_rgb1, pc_rgb2

        pc_1_try = rigid_transformation(r, t, pc2)

        #sse = sum( np.linalg.norm( pc_1_try[:, points_vect[1]]-pc1[:, points_vect[0]], axis = 0 ) )   # shape should be 1*307200
        for pt in range(len(xy[0])):
            num_inliers = 0
            sse = np.linalg.norm( pc_1_try[:, pos_vect( *xy[0][pt]) ] - pc1[:, pos_vect( *xy[1][pt]) ], axis = 0 )

            if sse < dist_inliers_threshold:
                num_inliers+=1

        if(num_inliers > percentage_inliers_threshold*len(xy[0])):
            print('Found a nice [RT] !! :)')
            r_best, t_best = (r, t)
            pc_try_best = pc_1_try
            break

        elif(max_inliers < num_inliers):

            r_best, t_best = (r, t)
            pc_try_best = pc_1_try

    else:
        r_best, t_best = (r, t)
        pc_try_best = pc_1_try

    print(r_best)
    print(t_best)

    return pc_try_best

def procrustes_nosso(pc1: np.array, pc2: np.array) -> ( tuple ):

    # http://printart.isr.tecnico.ulisboa.pt/piv/project/docs/Registration_explained.pdf

    pc1 = np.array([ [0,0,0], [1,1,1], [2,1,0],[1,2,0] ]).T
    pc2 = np.array([ [0,2,3], [1,3,2], [2,2,2],[1,2,1] ]).T

    '''
    RR = np.array([ [1,0,0], [0,0,1], [0,-1,0] ])
    TT = np.array([ [0,2,3] ])
    '''

    # setp 1: computes centroids

    centroid_1_vect = np.mean(pc1, axis=1)
    centroid_2_vect = np.mean(pc2, axis=1)

    # step 2: centers the point clouds

    centroid_1 = pc1 - np.expand_dims(centroid_1_vect, axis=1)
    centroid_2 = pc2 - np.expand_dims(centroid_2_vect, axis=1)

    # step 3: computes covariance matrix (3x3)

    cov_matrix = centroid_1@centroid_2.T

    # step 4: computes SVD

    # u, v = SVD(cov_matrix)
    u, h, v = np.linalg.svd(cov_matrix)

    # step 5: Rotation matrix

    R = u@np.array([[1, 0, 0],
                    [0, 1, 0 ], 
                    [0, 0, np.linalg.det(u@v)]])@v
    
    # setp 6: Translation vector

    T = centroid_1_vect - R.T@centroid_2_vect

    aa = R@pc1 + T.reshape(3,1)

    return (R, T.reshape(3,1))


def SVD( cov_matrix: np.array ) -> ( tuple ):
    """
    Computes the singular value decomposition of cov_matrix
    https://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm
    """

    # computes eigenvectores
    _, U = np.linalg.eig(cov_matrix@cov_matrix.T)    
    a, V = np.linalg.eig(cov_matrix.T@cov_matrix)


    c = U@np.diag(np.sqrt(a)@V.T)
    

    return (U, V)
   
def generate_depth_pc(dep1, k_depth):
    """
    Generates PointCould for the depth image
    """

    flat = lambda x: np.array(x, dtype=float).flatten('C')

    scale = flat(dep1)/1000

    nx = np.linspace(1, dep1.shape[1], dep1.shape[1], dtype=float)
    ny = np.linspace(1, dep1.shape[0], dep1.shape[0], dtype=float)
    [u, v] = np.meshgrid(nx, ny)

    z = np.ones( (scale.shape), dtype=float )

    depth_camera_frame_normalized = np.array([flat(u), flat(v), z])

    depth_camera_frame = np.array([line*scale for line in depth_camera_frame_normalized])

    xyz = np.linalg.inv(k_depth)@depth_camera_frame

    return xyz

def get_key_points(rgb1, rgb2):

    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(rgb1, mask = None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(rgb2, mask = None)

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

'''
Saves a matrix in mat format
'''
pc_in_matlab = lambda pc, name='pc': savemat( "pointcloud.mat", dict(zip(name, pc)) )

def rgb_pc_to_rgb_img(pc, k, img):

    dim = img.shape
    
    # rgb_frame = np.linalg.inv(k)@pc
    rgb_frame = k@pc

    u = np.divide( rgb_frame[0,:], rgb_frame[2,:] )
    v = np.divide( rgb_frame[1,:], rgb_frame[2,:] )

    # normalize coordinates
    u[ u < 0 ] = 0
    u[u > dim[1]-1] = dim[1]-1

    v[ v < 0 ] = 0
    v[ v > dim[0]-1 ] = dim[0]-1

    image = np.zeros(dim, dtype='uint8')
    img_ = np.reshape(img, (-1, 3) )

    for i in range( pc.shape[1] ):
    
        image[ round(v[i]) ][ round(u[i]) ] = img_[i]

    cv2.imshow('original',img)
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# página 3 do ficheiro, Part final
def batota(pc1, pc2):

    B = pc2;
    _, A, _ = procrustes(pc1, pc2)

    C = np.concatenate( (B, np.ones((1, B.shape[1]), dtype=float)) ).T

    try:
        RT = ( np.linalg.inv(C.T @ C) @ C.T @ A.T ).T
    except:
        return (np.zeros((3,3)), np.zeros((3,1)))

    r = RT[:,:3]
    t = RT[:,3:]

    return (r,t)
    
