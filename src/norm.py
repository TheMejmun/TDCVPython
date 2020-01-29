import numpy as np


def __zero_mean(image):
    return image - np.mean(image, axis=(0, 1))


def __unit_variance(image):
    std = (0., 0., 0.)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            print(image[y, x] ** 2)
            std += (image[y, x] ** 2)

    std /= image.shape[0] * image.shape[0]
    print('Standard deviation: ', std)
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
    print(image, '\n')

    print(__zero_mean(image), '\n')

    print(normalize(image))
