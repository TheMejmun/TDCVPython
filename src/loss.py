import numpy as np


def l_triplets(t, margin=0.01):
    out = 0
    for i in range(len(t) // 3):
        out += np.max(0,
                      1 -
                      (np.linalg.norm(t[i * 3] - t[i * 3 + 2]) ** 2) /
                      (np.linalg.norm(t[i * 3] - t[i * 3 + 1]) ** 2 + margin))
    return out


def l_pairs(t):
    out = 0
    for i in range(len(t) // 3):
        out += np.linalg.norm(t[i * 3] - t[i * 3 + 1]) ** 2
    return out
