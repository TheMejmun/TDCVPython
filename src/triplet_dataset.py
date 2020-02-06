from torch.utils.data import Dataset
import torch
import numpy as np
import random
from data import load_dataset


class TripletDataset(Dataset):
    def __init__(self, s_train=load_dataset('train'), s_db=load_dataset('db'), pusher_ratio=0.5):
        self.s_train = s_train
        self.s_db = s_db
        self.pusher_ratio = pusher_ratio

    def __len__(self):
        return len(self.s_train)

    def __getitem__(self, i):
        anchor = self.s_train[i]

        puller = self.__find_nn(anchor, self.s_db)

        # Depending on ratio, set pusher to same or different class
        # 0 -> none of same class
        # 1 -> all of same class
        if random.random() < self.pusher_ratio:
            pusher = random.choice([x for x in self.s_db if anchor[1] == x[1] and x[3] != puller[3]])
        else:
            pusher = random.choice([x for x in self.s_db if not anchor[1] == x[1]])

        return anchor[0], puller[0], pusher[0], anchor, puller, pusher

    def __find_nn(self, a, b_set):
        match = None
        match_d = float('inf')
        for b in b_set:
            # Same class only
            if a[1] == b[1]:

                # Angular distance
                # Formula given in exercise
                # Clip added to prevent NaNs
                d = self.__angular_distance(a[2], b[2])
                if d < match_d:
                    match = b
                    match_d = d

        return match

    @staticmethod
    def __angular_distance(a, b):
        return 2 * np.arccos(np.clip(np.abs(np.dot(a, b)), a_min=-1, a_max=1))
