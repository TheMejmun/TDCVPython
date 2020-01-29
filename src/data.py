from PIL import Image as I
import numpy as np
import glob
import re
import os


# Loads image as numpy tensor
# Converts to RGB mode just in case
def load_image(path):
    return np.array(I.open(path).convert('RGB'))


# Stores image
def store_image(image, path, format='PNG'):
    image = I.fromarray(image)
    image.save(path, format=format)


# Path MUST end in a slash
# Returns a list of tuples
# return[0] is raw image data
# return[1] is a tuple with given pose data
def load_folder(path):
    image_dict = {}

    # Get all files in path
    for f in glob.iglob(path + '*'):
        # Is image
        if re.match('.*\.png$', f):
            # Put images into dict with file name (without path) as index
            image_dict[f[len(path):]] = load_image(f)

    out = list()

    for f in glob.iglob(path + '*'):
        # Is poses.txt
        if re.match('.*poses\.txt$', f):

            lines = open(f, 'r').readlines()

            for i in range(len(lines) // 2):
                pose_strings = lines[i * 2 + 1].split()
                pose = (float(pose_strings[0]),
                        float(pose_strings[1]),
                        float(pose_strings[2]),
                        float(pose_strings[3]))

                out.append((image_dict[lines[i * 2][2:-1]],
                            pose))

    return out
