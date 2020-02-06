import numpy as np
import torch.functional as F


# If batch is set, switch to dynamic margin
def l_triplets(o_anchors, o_pullers, o_pushers, anchors=None, pushers=None, margin=0.01):
    out = 0
    for i in range(o_anchors.shape[0]):

        if anchors is not None and pushers is not None:
            # Passing tuples into function where [0] is the class and [1] is pose data
            margin = __dynamic_margin((anchors[1][i], anchors[2][i]), (pushers[1][i], pushers[2][i]))

        x = 1 - (F.norm(o_anchors[i] - o_pushers[i]) ** 2) / (F.norm(o_anchors[i] - o_pullers[i]) ** 2 + margin)
        if x > 0:
            out += x

    return out


def l_pairs(o_anchors, o_pullers):
    out = 0
    for i in range(o_anchors.shape[0]):
        out += F.norm(o_anchors[i] - o_pullers[i]) ** 2
    return out


# Proposed by
# Sergey Zakharov, Wadim Kehl, Benjamin Planche, Andreas Hutter, Slobodan Ilic
def __dynamic_margin(anchor, pusher, n=np.pi):
    # If same class
    if anchor[0] == pusher[0]:
        return 2 * np.arccos(np.clip(np.abs(np.dot(anchor[1], pusher[1])), a_min=-1, a_max=1))
    else:
        return n
