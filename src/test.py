import datetime
from time import time

import numpy as np
from matplotlib import pyplot as plt

from data import load_dataset
from nn import Net


def test(s_test, s_db, run=0, writer=None):
    start_t = time()
    print('\nTesting')

    # Load NN
    print('Loading NN')
    net = Net().double()
    # Load state dict
    net.load()
    net.eval()

    # Get results for everything in s_test and s_db
    print('Getting descriptors')
    results_test = list()
    for x in s_test:
        results_test.append(net(x[0].view(1, 3, 64, 64)).detach().numpy())
    results_db = list()
    for x in s_db:
        results_db.append(net(x[0].view(1, 3, 64, 64)).detach().numpy())

    # TODO replace with library
    print('Finding closest matches')
    matches = list()
    for i_x in range(len(s_test)):
        match = None
        match_d = float('inf')
        for i_y in range(len(s_db)):
            d = np.linalg.norm(results_db[i_y] - results_test[i_x])
            if d < match_d:
                match_d = d
                match = i_y

        matches.append((s_test[i_x], s_db[match]))

    print('Evaluating matches')
    # How many matches are within 10, 20, 40, and 180 degrees
    lt10 = lt20 = lt40 = lt180 = 0
    hist_list = list()
    confusion_matrix = {'ape': {'ape': 0, 'benchvise': 0, 'cam': 0, 'cat': 0, 'duck': 0},
                        'benchvise': {'ape': 0, 'benchvise': 0, 'cam': 0, 'cat': 0, 'duck': 0},
                        'cam': {'ape': 0, 'benchvise': 0, 'cam': 0, 'cat': 0, 'duck': 0},
                        'cat': {'ape': 0, 'benchvise': 0, 'cam': 0, 'cat': 0, 'duck': 0},
                        'duck': {'ape': 0, 'benchvise': 0, 'cam': 0, 'cat': 0, 'duck': 0}}
    for i in matches:

        # [real][predicted]
        confusion_matrix[i[0][1]][i[1][1]] += 1

        if i[0][1] == i[1][1]:

            # Angular distance
            # Formula given in exercise
            # Clip added to prevent NaNs
            angular_distance = 2 * np.arccos(np.clip(np.abs(np.dot(i[0][2], i[1][2])), a_min=-1, a_max=1))
            degrees = (angular_distance / np.pi) * 180

            hist_list.append(degrees)

            if degrees < 10:
                lt10 += 1
            if degrees < 20:
                lt20 += 1
            if degrees < 40:
                lt40 += 1
            if degrees < 180:
                lt180 += 1

    # Print histogram
    plt.hist(x=hist_list,
             bins=(0, 10, 20, 40, 180),
             range=(0, len(s_test)),
             cumulative=True)
    plt.show()

    # Print confusion matrix
    for real_label in ('ape', 'benchvise', 'cam', 'cat', 'duck'):
        print('\n', real_label[:3], end='\t')
        for predicted_label in ('ape', 'benchvise', 'cam', 'cat', 'duck'):
            print(str(np.around(confusion_matrix[real_label][predicted_label] / (len(s_test) / 5), 2))[1:], end='\t')
    print('\nre/pr\t', end='')
    for predicted_label in ('ape', 'benchvise', 'cam', 'cat', 'duck'):
        print(predicted_label[:3], end='\t')
    print('\n')

    if writer is not None:
        writer.add_scalar(tag='match_within_10_degrees',
                          scalar_value=lt10 / len(s_test),
                          global_step=run)
        writer.add_scalar(tag='match_within_20_degrees',
                          scalar_value=lt20 / len(s_test),
                          global_step=run)
        writer.add_scalar(tag='match_within_40_degrees',
                          scalar_value=lt40 / len(s_test),
                          global_step=run)
        writer.add_scalar(tag='match_within_180_degrees',
                          scalar_value=lt180 / len(s_test),
                          global_step=run)
    else:
        print()
        print('Matches within 10 degrees:\t', lt10, '/', len(s_test), '\tRatio:', lt10 / len(s_test))
        print('Matches within 20 degrees:\t', lt20, '/', len(s_test), '\tRatio:', lt20 / len(s_test))
        print('Matches within 40 degrees:\t', lt40, '/', len(s_test), '\tRatio:', lt40 / len(s_test))
        print('Matches within 180 degrees:\t', lt180, '/', len(s_test), '\tRatio:', lt180 / len(s_test))
        print()

    print('Finished in ', str(datetime.timedelta(seconds=time() - start_t)), 's\n')


if __name__ == '__main__':  # Only execute if called
    test(s_test=load_dataset('test'), s_db=load_dataset('db'))
