import numpy as np
import torch.functional as F


def l_triplets(t, margin=0.01):
    out = 0
    for i in range(len(t) // 3):
        x = 1 - (F.norm(t[i * 3] - t[i * 3 + 2]) ** 2) / (F.norm(t[i * 3] - t[i * 3 + 1]) ** 2 + margin)
        if x > 0: out += x

    return out


def l_pairs(t):
    out = 0
    for i in range(len(t) // 3):
        out += F.norm(t[i * 3] - t[i * 3 + 1]) ** 2
    return out
