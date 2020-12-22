import cv2
import sys
import scipy.io
import numpy as np
from image_relations import *
from weight_graph import *
import os

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv) )

# Run: python3  rigidtransforms.py  rgbimgs.txt depthimgs.txt cameracalib.txt transforms.txt

# rgbimgs.txt
with open(sys.argv[1], 'r') as file:

    rgbimgs = []
    for img in file.readlines():
        rgbimgs.append( cv2.imread("./"+img.strip("\n"), cv2.IMREAD_COLOR)) if img.split('.')[-1].strip("\n") != 'mat' else rgbimgs.append(scipy.io.loadmat(img.strip("\n"))["depth_array"] )

# depthimgs.txt
with open(sys.argv[2], 'r') as file:

    depthimgs = []
    for img in file.readlines():
        depthimgs.append( cv2.imread("./"+img.strip("\n"), cv2.IMREAD_GRAYSCALE)) if img.split('.')[-1].strip("\n") != 'mat' else depthimgs.append(scipy.io.loadmat(img.strip("\n"))["depth_array"] )

# cameracalib.txt
with open(sys.argv[3], 'r') as calib:

    k_rgb = np.fromfile(calib, dtype=float, sep=' ', count=9).reshape((3,3))
    k_depth = np.fromfile(calib, dtype=float, sep=' ', count=9).reshape((3,3))
    r_depth2rgb = np.fromfile(calib, dtype=float, sep=' ', count=9).reshape((3,3))
    t_depth2rgb = np.fromfile(calib, dtype=float, sep=' ', count=3).reshape((3,1))


# transforms.txt
with open(sys.argv[4], 'r') as file:
    
    path_file_name_output = file.readline()

# RT = []
# for i in range( len( rgbimgs )-1 ):
#     fix = i+1
#     if(i==fix): continue
#     print(f'Computing RT ... {(i+1)}/{len(rgbimgs)-1}', end='\t -> \t', flush=True)
#     camera = ( rgbimgs[i], rgbimgs[fix], depthimgs[i], depthimgs[fix], k_rgb,  k_depth, r_depth2rgb, t_depth2rgb )
#     R, T = transformation2cameras(camera, fix, i)

#     RT.append( np.concatenate( (R, T.T), axis=0).flatten() )

# np.savetxt(fname=path_file_name_output, 
#             X=RT,
#             delimiter='\t',
#             newline='\n')

# try:    os.remove("point_clouds.p")
# except: pass

# camera = ( rgbimgs, depthimgs, k_rgb,  k_depth, r_depth2rgb, t_depth2rgb )
# build_graph(camera)



# camera = ( rgbimgs[1], rgbimgs[2], depthimgs[1], depthimgs[2], k_rgb,  k_depth, r_depth2rgb, t_depth2rgb )
# R, T = transformation2cameras(camera, 1, 2)


file = open( "rt_graph.p", "rb" )
dict_pc = pk.load( file )
file.close()

RT = []

keys_dict = list(dict_pc.keys())
keys_dict.sort()

for i in keys_dict:
    R_total = np.identity(3)
    T_total = np.zeros((3,1))
    parent = i
    while(True):
        R_total = R_total@dict_pc[parent]["R"]
        T_total = T_total + dict_pc[parent]["T"]
        parent = dict_pc[parent]["parent"]
        if parent == 0:
            RT.append( np.concatenate( (R_total, T_total.T), axis=0).flatten() )
            break


np.savetxt(fname=path_file_name_output, 
            X=RT,
            delimiter='\t',
            newline='\n'
            )