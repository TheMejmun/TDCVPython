from time import time
import torch
from data import load_dataset
import numpy as np
from nn import Net
import datetime
from torch.utils.tensorboard import SummaryWriter


def test(run=0, s_test=None, s_db=None):
    start_t = time()
    print('\nTesting')

    writer = SummaryWriter('runs/eval')

    # Load data
    if s_test is None:
        s_test = load_dataset('train')
    if s_db is None:
        s_db = load_dataset('db')

    # Load NN
    print('Loading NN')
    net = Net().double()
    try:
        net.load_state_dict(torch.load('state_dict.pth'))
    except FileNotFoundError:
        print('State dict not found')
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
    class_match_count = 0
    # How many matches are within 10, 20, 40, and 180 degrees
    lt10 = lt20 = lt40 = lt180 = 0
    for i in matches:
        if i[0][1] == i[1][1]:
            class_match_count += 1

            # Angular distance
            # Formula given in exercise
            # Clip added to prevent NaNs
            angular_distance = 2 * np.arccos(np.clip(np.abs(np.dot(i[0][2], i[1][2])), a_min=-1, a_max=1))
            degrees = (angular_distance / np.pi) * 180
            if degrees < 10:
                lt10 += 1
            if degrees < 20:
                lt20 += 1
            if degrees < 40:
                lt40 += 1
            if degrees < 180:
                lt180 += 1

    writer.add_scalar(tag='class_matches',
                      scalar_value=class_match_count / len(s_test),
                      global_step=run)
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

    writer.close()
    print('Finished in ',  str(datetime.timedelta(seconds=time() - start_t)), 's\n')

