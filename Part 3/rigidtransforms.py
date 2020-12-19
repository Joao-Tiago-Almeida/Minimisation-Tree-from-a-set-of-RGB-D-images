import cv2
import sys
import scipy.io
import numpy as np
from image_relations import *

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
    
    file_name_output = file.readline()

# cv2.imshow('image', rgbimgs[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

camera = ( rgbimgs[0], rgbimgs[1], depthimgs[0], depthimgs[1], k_rgb,  k_depth, r_depth2rgb, t_depth2rgb )

transformation2cameras(camera)