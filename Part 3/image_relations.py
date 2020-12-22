import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange as rdr
from scipy.io import savemat
import pickle as pk


def transformation2cameras( camera: tuple, pc1_file: int = 0, pc2_file: int = 1 ) -> ( tuple ):
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

    depth_pc_2_rgb_pc = lambda xyz: np.concatenate( (r_depth2rgb, t_depth2rgb), axis=1)@np.concatenate( (xyz, np.ones((1, xyz.shape[1]), dtype=float) ), axis=0 )
    
    try:
        file = open( "point_clouds.p", "xb" )
        dict_pc = dict.fromkeys(range(0, 2))
    except:
        file = open( "point_clouds.p", "rb" )
        dict_pc = pk.load( file )
    file.close()

    
    if pc1_file not in dict_pc: # already does not exist in the file
        dict_pc[pc1_file] = None

    pc_rgb1 = dict_pc[pc1_file]

    if pc_rgb1 is None:
        pc_rgb1 = depth_pc_2_rgb_pc( generate_depth_pc(dep1, k_depth) )
        dict_pc[pc1_file] = pc_rgb1
    
    if pc2_file not in dict_pc:
        dict_pc[pc2_file] = None
    
    pc_rgb2 = dict_pc[pc2_file]

    if pc_rgb2 is None:
        pc_rgb2 = depth_pc_2_rgb_pc( generate_depth_pc(dep2, k_depth) )
        dict_pc[pc2_file] = pc_rgb2

    with open( "point_clouds.p", "wb" ) as file:
        pk.dump( dict_pc, file, protocol=pk.HIGHEST_PROTOCOL )

    if( len(xy[0]) ) < 150:
        return ( None, None, 0 )

    r, t, ratio_inliers, pc_new = ransac(pc_rgb1, pc_rgb2, xy, rgb1.shape)

    pc_in_matlab( (pc_rgb1, pc_new), (rgb1, rgb1), ('PC1', 'PC2') )

    #rgb_pc_to_rgb_img( pc_new, k_rgb, rgb1 )  #remove

    return (r, t, ratio_inliers)

def ransac( pc1: np.array, pc2: np.array, xy: list, dim: tuple, percentage_inliers_threshold = 0.5 ) -> ( tuple ):
    
    num_itr = len(xy[0])//4
    # percentage_inliers_threshold = 0.5  # percentage
    dist_inliers_threshold = 0.2    # meters

    # best values
    r_best = np.zeros((3,3), dtype=float)
    t_best = np.zeros((3,1), dtype=float)
    dist_best = np.Inf
    pc_try_best = None
    max_inliers = 0
    best_inliers = [[], []]
    already_nice_rt = False
    max_inliers_tuned = 0

    pos_vect = lambda x, y: ( round(x)*dim[0] + round(y) ) if dim[0] > dim[1] else ( round(x) + round(y)*dim[1] )
    pos_vect_inv = lambda idx: ( idx//dim[0], idx%dim[0] ) if dim[0] > dim[1] else ( idx%dim[1], idx//dim[1] )
    
    #rigid_transformation = lambda r, t, pc: np.concatenate((r, t), axis=1)@np.concatenate( (pc, np.ones((1, pc.shape[1]), dtype=float)), axis=0 )
    rigid_transformation = lambda r, t, pc: r@pc + t

    for i in range(num_itr):

        points = 4
        points_vect = [[],[]]

        for i in range(points):

            random_point = rdr( 0, len( xy[0] ) )   # bicates 'aleatori' point

            idx_1 = pos_vect( *xy[0][random_point] )
            idx_2 = pos_vect( *xy[1][random_point] )

            points_vect[0].append(idx_1)
            points_vect[1].append(idx_2)        

        r, t = procrustes(pc1[:, points_vect[0]], pc2[:, points_vect[1]])   # pc_rgb1, pc_rgb2

        pc_1_try = rigid_transformation(r, t, pc2)

        num_inliers = 0
        inliers = [[], []]
        for pt in range(len(xy[0])):    # count number of inliers
            dist = np.linalg.norm( pc1[:, pos_vect( *xy[0][pt]) ] - pc_1_try[:, pos_vect( *xy[0][pt]) ], axis = 0 )

            if dist < dist_inliers_threshold:    # check if it is an inlier
                inliers[0].append( pos_vect(*xy[0][pt]) )
                inliers[1].append( pos_vect(*xy[1][pt]) )
                num_inliers+=1

        if(num_inliers == 0): continue

        # tuning model - adjust RT based on point cloud already found
        r_tuning, t_tuning = procrustes( pc1[:, [inliers[0], inliers[1]][0]], pc_1_try[:, [inliers[0], inliers[1]][0]] )

        r_tuned = r_tuning @ r
        t_tuned = t_tuning + t

        pc_tuned = rigid_transformation( r_tuned, t_tuned, pc2 )

        num_inliers_tuned = check_num_inlers(pc1, pc_tuned, xy, dist_inliers_threshold, pos_vect)

        if(num_inliers_tuned > num_inliers):    # checks if tuning improves the transformation
            r = r_tuned
            t = t_tuned
            pc_1_try = pc_tuned
            num_inliers = num_inliers_tuned

        if(num_inliers > percentage_inliers_threshold*len(xy[0])):  # found a nice rigid transform

            r_best, t_best = (r, t)
            pc_try_best = pc_1_try
            max_inliers = num_inliers
            best_inliers = [inliers[0], inliers[1]]
            best_inliers = best_inliers.copy()
            already_nice_rt = True
            max_inliers_tuned = max(num_inliers_tuned, max_inliers_tuned)
            break

        elif(max_inliers < num_inliers):    # updates the best rigid transform

            best_inliers = [inliers[0], inliers[1]]
            best_inliers = best_inliers.copy()
            max_inliers = num_inliers
            r_best, t_best = (r, t)
            pc_try_best = pc_1_try
            max_inliers_tuned = max(num_inliers_tuned, max_inliers_tuned)


    print(f'Number of inliers ransac {max_inliers}', end='\t >> \t')

    has_tuned = '\t:(' if max_inliers_tuned == max_inliers else ':)\t'

    print(f'Tuning: {max_inliers_tuned}\t{has_tuned}\t{"[ *** Found a nice [RT] !! :) *** ]" if already_nice_rt else ""}\t', flush=True)

    ratio_inliers = max_inliers/len(xy[0])

    return ( r_best, t_best, ratio_inliers, pc_try_best ) 

def check_num_inlers( pc1: np.array, pc2: np.array, xy: list, dist_inliers_threshold: int, f_map) ->( int ):

    num_inliers = 0
    for pt in range(len(xy[0])):
        dist = np.linalg.norm( pc1[:, f_map( *xy[0][pt]) ] - pc2[:, f_map( *xy[0][pt]) ], axis = 0 )

        if dist < dist_inliers_threshold:
            num_inliers+=1

    return num_inliers

def procrustes(pc1: np.array, pc2: np.array) -> ( tuple ):

    # http://printart.isr.tecnico.ulisboa.pt/piv/project/docs/Registration_explained.pdf
    '''
    pc2 = np.array([ [0,0,0], [1,1,1], [2,1,0],[1,2,0] ]).T
    pc1 = np.array([ [0,2,3], [1,3,2], [2,2,2],[1,2,1] ]).T
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

    #U, V = SVD(cov_matrix)
    u, h, v = np.linalg.svd(cov_matrix)

    # step 5: Rotation matrix

    R = u@np.array([[1, 0, 0],
                    [0, 1, 0 ], 
                    [0, 0, np.linalg.det(u@v)]])@v
    
    # setp 6: Translation vector

    T = centroid_1_vect - R@centroid_2_vect

    return (R, T.reshape(3,1))

# pay attention because the eigenvectores are auto sign-normalized
def SVD( cov_matrix: np.array ) -> ( tuple ):
    """
    Computes the singular value decomposition of cov_matrix
    https://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm
    """

    # computes eigenvectores
    _, U = np.linalg.eig(cov_matrix@cov_matrix.T)
    _, V = np.linalg.eig(cov_matrix.T@cov_matrix)
    
    return (U, V.T)
   
def generate_depth_pc(dep1, k_depth):
    """
    Generates PointCould for the depth image
    """

    flat = lambda x: np.array(x, dtype=float).flatten( 'C' if dep1.shape[1] > dep1.shape[0] else 'F' )

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
def pc_in_matlab( pc, rgb, name='pc'):
    
    if not isinstance(rgb, tuple):
        rgb = (rgb,)
    
    if not isinstance(rgb, tuple):
        pc = (pc,)

    new_rgb = np.zeros_like( pc[0], dtype='uint8')

    flat = lambda x: np.array(x, dtype=float).flatten( 'C' if new_rgb.shape[1] > new_rgb.shape[0] else 'F' )

    pc_colored = []
    for rgb_, pc_ in zip(rgb, pc): # all images
        for i in range(3):  # RGB
            new_rgb[i,:] = flat(rgb_[:,:,i])

        pc_colored.append( np.concatenate((pc_, new_rgb)) )

    savemat( "pointcloud.mat", dict(zip(name, pc_colored) ) )

def rgb_pc_to_rgb_img(pc, k, img):

    dim = img.shape
    
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


    cv2.imshow('imagem 1',img)
    cv2.imshow('pata',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

