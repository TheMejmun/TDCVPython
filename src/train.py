from time import time
import torch
from data import load_dataset

if __name__ == '__main__':  # Only execute if called
    start_t = time()
    print('Training')

    print('CUDA is available' if torch.cuda.is_available() else 'CUDA is NOT available')

    s_train = load_dataset('train')
    s_db = load_dataset('db')

    print('Finished in ', round(time() - start_t, 2), 's')
