import cv2
from homography2cameras import *

# Video camera properties
pathInputVideo = "videos/NotSoNasty.mp4"
cap = cv2.VideoCapture(pathInputVideo)

final_H = None #pk.load(open('HT.p', 'rb'))
#reading image

template_file = "images/template2.png"

template = cv2.imread(template_file)

resize = lambda x, scale=2: cv2.resize(x, (int(x.shape[1]/scale) , int(x.shape[0]/scale) ))

template = resize(template)
gen = infinite_sequence()
best_H = None
max_inliers = 0 # minimum numer
frame_count = 0

while True:

    if cap.isOpened():
    
        ret, img = cap.read()

        if not ret:
            break

        print(f'\r Frame: {next(gen)}', end ='')

        img = resize(img)

        if (frame_count == 0): final_H = getHomography( img, template )
        
        if isinstance(final_H,list):
            max_inliers, best_H  = tuple(final_H)  if max_inliers < final_H[0] else (max_inliers, best_H)
            frame_count = 1
            final_H = None
        
        frame_count = frame_count+1 if (frame_count<=25 and frame_count>0) else 0  # waits 1 second when can not find a proper matrix

        if final_H is not None:
            break

cap.release()

final_H = final_H if final_H is not None else best_H

if final_H is not None:

    print("Recording video")
    frame_array=[]
    width = template.shape[1]
    height = template.shape[0]
    cap = cv2.VideoCapture(pathInputVideo)

    while True:
    
        if cap.isOpened():

            ret, img = cap.read()

            if not ret:
                break

            img = resize(img)

            frame_array.append(cv2.warpPerspective(img, final_H, (width,height)))


    size = (frame_array[0].shape[1],frame_array[0].shape[0])

    is_colored = frame_array[0].shape[2] == 3

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    pathOutputVideo = pathInputVideo.replace(".mp4","_output.mp4")

    out = cv2.VideoWriter(pathOutputVideo,fourcc, 24.0, size, is_colored) 


    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])

    out.release()

else:
    print("Could not find any H matrix ! :( ")