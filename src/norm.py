import numpy as np


def __zero_mean(image):
    mean=np.mean(image, axis=(0, 1))
    # print('Mean: ', mean)
    return image - mean


def __unit_variance(image):
    std = np.std(image, axis=(0, 1))
    # print('Standard deviation: ', std)
    return image / std


def normalize(image):
    return __unit_variance(__zero_mean(image))


# TESTING
if __name__ == '__main__':  # Only execute if called

    image = np.array(
        [[[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]],

         [[1, 1, 1],
          [2, 2, 2],
          [3, 3, 3]],

         [[1, 2, 3],
          [1, 2, 3],
          [1, 2, 3]]]
    )