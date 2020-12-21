import cv2
import sys
import scipy.io
import numpy as np
from image_relations import *
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

RT = []

for i in range( len( rgbimgs ) - 1 ):
    print(f'Computing RT ... {(i+1)}/{len(rgbimgs)-1}', end='\t -> \t', flush=True)
    camera = ( rgbimgs[i], rgbimgs[i+1], depthimgs[i], depthimgs[i+1], k_rgb,  k_depth, r_depth2rgb, t_depth2rgb )
    R, T = transformation2cameras(camera, i, i+1)
    RT.append( np.concatenate( (R, T.T), axis=0).flatten() )

np.savetxt(fname=path_file_name_output, 
            X=RT,
            delimiter='\t',
            newline='\n')

try:    os.remove("point_clouds.p")
except: pass

# camera = ( rgbimgs[1], rgbimgs[2], depthimgs[1], depthimgs[2], k_rgb,  k_depth, r_depth2rgb, t_depth2rgb )
# R, T = transformation2cameras(camera, 1, 2)