import os

def load_images_from_folder(folder):

    filetxt = ('rgbimgs.txt', 'depthimgs.txt')
    directory = ('rgb', 'depth')
    extension = ('png', 'jpeg', 'mat', 'jpg')

    for txt_, dir_ in zip(filetxt, directory):

        f = open(txt_, 'w')
        images = []

        for filename in os.listdir(f'{folder}/{dir_}'):

            if(filename.split('.')[-1].lower() in extension):
                images.append(f'{folder}/{dir_}/{filename}\n')
    

        images.sort(key = lambda filename : int(filename.split('/')[-1].split('_')[-1].split('.')[0]))

        for file in images:
            f.write(file)

        f.close()


folder = "newpiv2"

load_images_from_folder(folder)
