# -*- coding: utf-8 -*-
"""
Created on Sun May  3 19:37:14 2020

"""

from libtiff import TIFF, TIFFfile
import numpy as np
import imageio
import os
import os.path

# Transform the tiff file to frame images
def tiff_to_image_array(tiff_image_name, out_folder, out_type):
    '''
    tiff_image_name: string, the name of the tiff file
    out_folder: path for the generated images
    out_type:  the type of generated images
    '''
    tif = TIFF.open(tiff_image_name, mode = "r")
    idx = 0
    for im in list(tif.iter_images()):
        #
        im_name = out_folder  + str(idx) + out_type
        imageio.imwrite(im_name, im)
        print (im_name, 'successfully saved!')
        idx = idx + 1
    return


def create_gif(gif_name, path, duration):
    '''
    gif_name: string, the name of the generated gif file with the suffix '.gif'
    path: path for the generated gif
    duration :  time interval between the consecutive frames
    '''
    frames = []
    pngFiles = os.listdir(path)
    pngFiles.sort(key=lambda x:int(x[:-4]))
    image_list = [os.path.join(path, f) for f in pngFiles]
    for image_name in image_list:
        print(image_name)
        frames.append(imageio.imread(image_name))

    imageio.mimwrite(gif_name, frames, 'GIF', duration = duration)

    return
