import cv2
from homography import *

def resize(image):
    return cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))

def part1(template_image, moviein_filename, movieout_filename):
    # Get template and resize it to speed up the transformations
    template = cv2.imread(template_image)
    template = resize(template)

    best_H = None
    homography = None
    inliers_percentage = 0 
    frame_count = 0

    # Start video 
    cap = cv2.VideoCapture(moviein_filename)
    if (cap.isOpened()== False):  # Error checking
        print("We were unable to open the provided video file")
    
    # Read every frame of the video
    while(cap.isOpened()):
        # Get the current frame
        ret, img = cap.read()
        if ret == False:
            break

        img = resize(img)
        if (frame_count == 0):
            homography = get_homography(img, template)
        
        # If the homography doesn't make the thresholds we get a list
        if isinstance(homography, list):
            inliers_percentage, best_H = tuple(homography) if inliers_percentage < homography[0] else (inliers_percentage, best_H)
            frame_count = 1
            homography = None 
        
        # Wait a second when a matrix is not found. 1 second = 24 frames
        frame_count = frame_count+1 if (frame_count<24 and frame_count>0) else 0

        if homography is not None: # Once we find a homography that meets the requirements, stop
            break
    
    # Once we read the whole input video, close it
    cap.release()

    homography = homography if homography is not None else best_H
    if homography is None:
        print('Error geting a homography matrix for your video')
        exit()
    
    height = template.shape[0]
    width = template.shape[1]

    frames = []
    cap = cv2.VideoCapture(moviein_filename)
    if (cap.isOpened()== False):  # Error checking
        print("We were unable to open the provided video file")
    
    # Read every frame of the video
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == False:
            break

        img = resize(img)
        frames.append(cv2.warpPerspective(img, homography, (width,height)))
    
    cap.release()

    out = cv2.VideoWriter(filename = movieout_filename, fourcc = cv2.VideoWriter_fourcc(*'mp4v'), fps = 24.0, frameSize = (frames[0].shape[1],frames[0].shape[0]), isColor = frames[0].shape[2] == 3)

    for i in range(len(frames)):
        out.write(frames[i])
    out.release()

template_image = 'images/templateP1.png'
moviein_filename = 'videos/video1.mp4'
movieout_filename = moviein_filename.replace(".mp4",".output.mp4")
part1(template_image,moviein_filename,movieout_filename)