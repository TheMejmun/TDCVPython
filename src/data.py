import glob
import os
import re

import numpy as np
import torch
from PIL import Image as I

from norm import normalize

# TODO put dataset folder into project root
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


# Loads image as numpy tensor
# Converts to RGB mode just in case
def __load_image(path):
    return np.array(I.open(path).convert('RGB'))


# folder is one of the following: 'coarse', 'fine', or 'real'
# class_name is the name of the subfolder ('ape', 'cat',...)
#
# Returns a list of tuples
# return[0] is normalized image data
# return[1] is the class name
# return[2] is a np array with given pose data
# return[3] is file name with path
def __load_folder(folder, class_name):
    image_dict = dict()
    path = ROOT_DIR + '\\dataset\\' + folder + '\\' + class_name + '\\'

    # Get all files in path
    for f in glob.iglob(path + '*'):
        # Is image
        if re.match('.*\.png$', f):
            # Put images into dict with file name (without path) as index. Images get normalized, then converted to
            # PyTorch tensors, then permuted to (Channels, Height, Width)
            image_dict[f[len(path):]] = torch.from_numpy(normalize(__load_image(f))).permute((2, 0, 1))

    out = list()

    for f in glob.iglob(path + '*'):
        # Is poses.txt
        if re.match('.*poses\.txt$', f):

            lines = open(f, 'r').readlines()

            for i in range(len(lines) // 2):
                pose_strings = lines[i * 2 + 1].split()
                pose = np.array((float(pose_strings[0]),
                                 float(pose_strings[1]),
                                 float(pose_strings[2]),
                                 float(pose_strings[3])))

                out.append((image_dict[lines[i * 2][2:-1]],
                            class_name,
                            pose,
                            path + lines[i * 2][2:-1]))

    return out


# set mode to 'all', 'test', 'train', or 'db'
# 'all' returns a dict of all datasets
# everything else returns a list of tuples where:
# return[0] is normalized image data
# return[1] is the class name
# return[2] is a np array with given pose data
# return[3] is file name with path
def load_dataset(mode='all'):
    classes = ['ape', 'benchvise', 'cam', 'cat', 'duck']
    # Split string with ', ' and put casted ints into list
    training_split = [int(i) for i in open(ROOT_DIR + '\\dataset\\real\\training_split.txt').read().split(sep=', ')]

    if mode == 'all':
        print('Loading all datasets')
        return {'train': load_dataset('train'),
                'test': load_dataset('test'),
                'db': load_dataset('db')}

    elif mode == 'train':
        print('Loading train dataset')

        # Load all from 'fine' folder
        out = list()
        for c in classes:
            out.extend(__load_folder('fine', c))

        # Only put data from 'real' into training set, if part of training_split
        for c in classes:
            real_data = __load_folder('real', c)

            for i in range(len(real_data)):
                if i in training_split:
                    # print('Appending image with index ', i, ' from ', c, ' dataset to training set')
                    out.append(real_data[i])
        return out

    elif mode == 'test':
        print('Loading test dataset')

        out = list()
        # Only put data from 'real' into training set, if NOT part of training_split
        for c in classes:
            real_data = __load_folder('real', c)

            for i in range(len(real_data)):
                if i not in training_split:
                    # print('Appending image with index ', i, ' from ', c, ' dataset to testing set')
                    out.append(real_data[i])
        return out

    elif mode == 'db':
        print('Loading database dataset')

        # Load all from 'coarse' folder
        out = list()
        for c in classes:
            out.extend(__load_folder('coarse', c))
        return out

    else:
        print('Error: Unknown dataset loading mode')


# TESTING
if __name__ == '__main__':  # Only execute if called
    db = load_dataset('db')
    print()
