from time import time
import os

dirname = os.path.dirname
ROOT_DIR = dirname(dirname(__file__))

if __name__ == '__main__':  # Only execute if called
    start_t = time()

    print('Hello World from ', ROOT_DIR)

    print('Finished in ', round(time() - start_t, 2), 's')
