import cv2
from homography2cameras import *

# Video camera properties
pathInputVideo = "videos/video1.mp4"
cap = cv2.VideoCapture(pathInputVideo)

final_H = None

#reading image

template_file = "images/template2.png"

template = cv2.imread(template_file)

resize = lambda x: cv2.resize(x, (int(x.shape[1]/2) , int(x.shape[0]/2) ))

template = resize(template)

while True:

    if cap.isOpened():
    
            ret, img = cap.read()

            if not ret:
                break

            img = resize(img)

            final_H = getHomography( img, template )

            if final_H is not None:
                break

cap.release()


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