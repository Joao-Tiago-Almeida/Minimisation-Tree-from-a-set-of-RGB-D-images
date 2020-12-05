import os

def load_images_from_folder(folder, filetxt, extension):

    f=open(filetxt, 'w')

    images = []
    for filename in os.listdir(folder):
        if(filename[-3:]==extension):
            images.append(f'{folder}/{filename}\n')
    

    images.sort(key = lambda filename : int(filename.split('_')[-1].split('.')[0]))

    for file in images:
        f.write(file)
    f.close()

    return images

folder = "rbg_image"
rgb_filename = 'rgbimgs.txt'

load_images_from_folder(folder, rgb_filename, 'png')

folder = "depth_imgs"
depth_filename = 'depthimgs.txt'

load_images_from_folder(folder, depth_filename, 'mat')