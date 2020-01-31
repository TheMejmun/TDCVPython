import numpy as np


def generate_triplet_batch(s_train, s_db, batch_size):
    batch = list()
    for i in range(batch_size):
        # Get random element from s_train as anchor
        anchor = s_train[np.random.randint(len(s_train))]

        puller = __find_nn(anchor, s_db)

        pusher = puller
        # Do until file names are no longer equal
        while pusher[3] == puller[3]:
            # print('pusher == puller')
            pusher = s_db[np.random.randint(len(s_db))]

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
            d = 2 * np.arccos(np.clip(np.abs(np.dot(b[2], a[2])), a_min=-1, a_max=1))
            if d < match_d:
                match = b
                match_d = d

    return match


# TESTING
if __name__ == '__main__':  # Only execute if called
    from data import load_dataset, __load_image
    from PIL import Image

    s_test = load_dataset('test')
    s_db = load_dataset('db')
    batch = generate_triplet_batch(s_test, s_db, 1)

    Image.fromarray(__load_image(batch[0][3])).show(title='Anchor')
    Image.fromarray(__load_image(batch[1][3])).show(title='Puller')
    Image.fromarray(__load_image(batch[2][3])).show(title='Pusher')
