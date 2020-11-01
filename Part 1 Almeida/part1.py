'''
Project ist190119
PIV project part1 -
inputs:

template_image - string with the path of the template image

moviein_filename - string with the input video

movieout_filename - string with the output video

return vars: None
'''

import cv2
from homography2cameras import *

DEBUG = True

def part1(template_image: str, moviein_filename: str, movieout_filename: str) -> (None):

    resize = lambda img, scale=2: cv2.resize(img, (int(img.shape[1]/scale) , int(img.shape[0]/scale) ))
    
    # Template properties
    template = cv2.imread(template_image)
    template = resize(template)

    # Variables declaration
    if DEBUG: gen = infinite_sequence()
    homography = None
    close_H = None
    ratio_inliers = 0
    frame_counter = 0

    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(moviein_filename)

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, img = cap.read()
        if ret == True:
            
            # Actual frame number
            if DEBUG: print(f'\rComputing homograpgy for frame {next(gen)+1}', end ='')

            # Resize the image and compute an homography
            img = resize(img)
            if (frame_counter == 0): homography = getHomography( frame = img,
                                                                 template = template,
                                                                 tresh_num_inliers = 0.50,
                                                                 tresh_dist_inliers = 1.5,
                                                                 max_iterations = 1000,
                                                                 min_inliers = 40,
                                                                 debug = DEBUG)

            # Get a list when the homography doesn't fulfil the requirements
            if isinstance(homography,list):
                ratio_inliers, close_H  = tuple(homography) if ratio_inliers < homography[0] else (ratio_inliers, close_H)
                frame_counter = 1
                homography = None

            # Wait 1 second (24 frames) when can not find a proper matrix
            frame_counter = frame_counter+1 if (frame_counter<24 and frame_counter>0) else 0

            # Exit when a finds a Homography that fulfil the requirements 
            if homography is not None:
                break
            
            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
 
        # Break the loop
        else: 
            break

    # When everything done, release the video capture object
    cap.release()
    print('')

    # Storage the Homography with the best found 
    homography = homography if homography is not None else close_H.copy()
    if homography is None:
        print("Could not find any H matrix ! :( - No output video")
        exit()
    
    # Output dimension
    height = template.shape[0]
    width = template.shape[1]

    # Auxiliar variables
    total_frames = next(gen)
    frame_number = 0

    # Declare an empty frame to store all the frames
    frames_sequence=[]

    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(moviein_filename)

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
     # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, img = cap.read()
        if ret == True:

            # Actual frame number
            if DEBUG:
                frame_number+=1
                print(f'\rComputing new prespective { "."*((frame_number%3)+1) }', end ='')

            # Resize the image and compute an homography
            img = resize(img)
            frames_sequence.append(cv2.warpPerspective(img, homography, (width,height)))

            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
 
        # Break the loop
        else: 
            break

    # When everything done, release the video capture object
    cap.release()
    print('')

    # Define the codec and create VideoWriter object.The output is stored in movieout_filename.
    # Define the fps to be equal to 24. Also frame size is passed.
    
    out = cv2.VideoWriter( filename = movieout_filename,
                           fourcc = cv2.VideoWriter_fourcc(*'mp4v'),
                           fps = 24.0, 
                           frameSize = (frames_sequence[0].shape[1],frames_sequence[0].shape[0]),
                           isColor = frames_sequence[0].shape[2] == 3 )
    
    # Read until video is completed
    for i in range( len( frames_sequence ) ):

        # Actual frame number
        if DEBUG: print(f'\rWriting video {100*(i+1)/len(frames_sequence):.2f}%', end ='')

        out.write(frames_sequence[i])

    out.release()


template_image = 'templates/templateP1.png'
moviein_filename = 'videos/Nasty.mp4'
movieout_filename = moviein_filename.replace(".mp4",".output.mp4")
part1(template_image,moviein_filename,movieout_filename)