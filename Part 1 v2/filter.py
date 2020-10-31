import cv2
import numpy as np
import pickle as pk


def median_rgb( rgb: np.array ) -> ( np.array ):
    temp = np.mean(
        a = rgb,
        axis = 0
    )
    return np.int_(temp)
        
def median_2D( frame: np.array ) -> ( np.array ):

    frame_new = frame.copy()    # create new frame

 
    for y in range( frame.shape[0] ):
        for x in range( frame.shape[1] ):

            median_list = []

            # search: clockwise

            # upper border
            if y == 0:
                if x == 0:  # upper left corner
                    median_list = [ frame[y][x], frame[y][x+1], frame[y+1][x+1], frame[y+1][x] ]

                elif x == frame.shape[1]-1: # upper right corner
                    median_list = [ frame[y][x], frame[y][x-1], frame[y+1][x-1], frame[y+1][x] ]

                else: 
                    median_list = [ frame[y][x], frame[y][x+1], frame[y+1][x+1], frame[y+1][x], frame[y+1][x-1], frame[y][x-1] ]

            # bottom border
            elif y == frame.shape[0]-1:
                if x == 0:  # bottom left corner
                    median_list = [ frame[y][x], frame[y][x+1], frame[y-1][x+1], frame[y-1][x] ]

                elif x == frame.shape[1]-1: # bottom right corner
                    median_list = [ frame[y][x], frame[y][x-1], frame[y-1][x-1], frame[y-1][x] ]

                else: 
                    median_list = [ frame[y][x], frame[y][x-1], frame[y-1][x-1], frame[y-1][x], frame[y-1][x+1], frame[y][x+1] ]
            
            # left border
            elif x == 0:
                median_list = [ frame[y][x], frame[y-1][x], frame[y-1][x+1], frame[y][x+1], frame[y+1][x+1], frame[y+1][x] ]

            # right border
            elif x == frame.shape[1]-1:
                median_list = [ frame[y][x], frame[y+1][x], frame[y+1][x-1], frame[y][x-1], frame[y-1][x-1], frame[y-1][x] ]

            # generic
            else:
                median_list = [ frame[y][x], frame[y-1][x], frame[y-1][x+1], frame[y][x+1], frame[y+1][x+1],
                                frame[y+1][x], frame[y+1][x-1], frame[y][x-1], frame[y-1][x-1] ]

            breakpoint
            frame_new[y][x] = median_rgb(median_list)

            breakpoint
    
    return frame_new

'''
frame = pk.load(open('frame.p', 'rb'))

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

filtered_image = median_2D(frame)

cv2.imshow('filtered_image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''