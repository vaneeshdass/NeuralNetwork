import glob
import os
import os

import cv2
import numpy as np
import pandas as pd

from PIL import Image


def get_img_list(in_dir, ext, is_srt=True):
    src_dir = in_dir + '/*.' + ext
    list_images = glob.glob(src_dir)
    if is_srt:
        list_images.sort()
    return list_images

def get_file_number(file_name):
    file_dir, file_name = os.path.split(file_name)
    file_no, file_ext = os.path.splitext(file_name)
    return file_no

def get_file_name(file_path):
    file_dir, file_name = os.path.split(file_path)
    return file_name


def absoluteFilePaths(directory):
   dirs = next(os.walk(directory))[1]
   for x in range(0, dirs.__len__()):
       dirs[x] = directory+dirs[x]
   return dirs



def make_dataframe_from_images(parent_dir_of_all_classess):
    dirs = absoluteFilePaths(parent_dir_of_all_classess)
    width = 50
    height = 50
    dim = (width, height)
    images_x = []
    images_y = []  # for labels
    for dir in dirs:
        for image in get_img_list(dir, ext='png'):
            imageMatrix = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            # resize image
            resized = cv2.resize(imageMatrix, dim, interpolation=cv2.INTER_AREA)
            image_row_vector = resized.reshape(-1)
            images_x.append(image_row_vector)
            images_y.append(int(get_file_name(dir)))

        # np.array(images_x.shape)
        df_images = pd.DataFrame(images_x)
        df_labels = pd.DataFrame(images_y)
        print('Done')

    return df_images,df_labels


def convert_image_to_given_format(in_dir,out_dir,source_image_format,format_to_convert):

    all_images = get_img_list(in_dir, ext=source_image_format)
    counter = 0
    for image in all_images:
        im = Image.open(image)
        im.save(out_dir + str(counter) + format_to_convert)
        counter = counter + 1

    print('all images converted to given format')


