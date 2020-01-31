import numpy as np
import torch.functional as F


# If s_train is set, switch to dynamic margin
def l_triplets(t, s_train=None, margin=0.01):
    out = 0
    for i in range(len(t) // 3):

        if s_train is not None:
            margin = __dynamic_margin(s_train[i * 3], s_train[i * 3 + 2])
            print('Dynamic margin: ', margin)

        x = 1 - (F.norm(t[i * 3] - t[i * 3 + 2]) ** 2) / (F.norm(t[i * 3] - t[i * 3 + 1]) ** 2 + margin)
        if x > 0: out += x

    return out


def l_pairs(t):
    out = 0
    for i in range(len(t) // 3):
        out += F.norm(t[i * 3] - t[i * 3 + 1]) ** 2
    return out


# Proposed by
# Sergey Zakharov, Wadim Kehl, Benjamin Planche, Andreas Hutter, Slobodan Ilic
def __dynamic_margin(anchor, pusher, n=2 * np.pi):
    # If same class
    if anchor[1] == pusher[1]:
        return 2 * np.arccos(np.clip(np.abs(np.dot(anchor[2], pusher[2])), a_min=-1, a_max=1))
    else:
        return n
