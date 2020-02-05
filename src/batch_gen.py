import numpy as np
import random


def generate_triplet_batch(s_train, s_db, batch_size, pusher_same_class_ratio=0.85):
    batch = list()

    for i in range(batch_size):
        # Get random element from s_train as anchor
        anchor = random.choice(s_train)

        puller = __find_nn(anchor, s_db)

        # Depending on input ratio and iteration, set pusher to same or different class
        if i < int(batch_size * pusher_same_class_ratio):
            pusher = random.choice([x for x in s_db if anchor[1] == x[1]])
        else:
            pusher = random.choice([x for x in s_db if not anchor[1] == x[1]])

        batch.extend((anchor, puller, pusher))
    return batch


def __find_nn(a, b_set):
    match = None
    match_d = float('inf')
    for b in b_set:
        # Same class only
        if a[1] == b[1]:

            # Angular distance
            # Formula given in exercise
            # Clip added to prevent NaNs
            d = __angular_distance(a[2], b[2])
            if d < match_d:
                match = b
                match_d = d

    return match


def __angular_distance(a, b):
    return 2 * np.arccos(np.clip(np.abs(np.dot(a, b)), a_min=-1, a_max=1))


# TESTING
if __name__ == '__main__':  # Only execute if called
    from data import load_dataset, __load_image
    from PIL import Image

    s_test = load_dataset('test')
    s_db = load_dataset('db')
    batch = generate_triplet_batch(s_test, s_db, 6)

    Image.fromarray(__load_image(batch[0][3])).show(title='Anchor')
    Image.fromarray(__load_image(batch[1][3])).show(title='Puller')
    Image.fromarray(__load_image(batch[2][3])).show(title='Pusher')

    # print(2 * np.arccos(1))
    # print(2 * np.arccos(-1))
