from time import time
import torch
from data import load_dataset
from batch_gen import generate_triplet_batch
from nn import Net
from loss import *
import torch.optim as optim
import datetime

EPOCHS = 1000
BATCH_SIZE = 100

if __name__ == '__main__':  # Only execute if called
    start_t = time()
    print('Training')

    print('CUDA is available' if torch.cuda.is_available() else 'CUDA is NOT available')

    # Load data
    s_train = load_dataset('train')
    s_db = load_dataset('db')

    # Load NN
    net = Net().double()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        epoch_start = time()

        print('\nEpoch: ', epoch)

        batch = generate_triplet_batch(s_train, s_db, BATCH_SIZE)
        results = list()

        optimizer.zero_grad()

        for i in batch:
            results.append(net(i[0].view(1, 3, 64, 64)))

        loss = l_triplets(results) + l_pairs(results)
        print('Loss: ', float(loss))

        loss.backward()
        optimizer.step()

        torch.save(net.state_dict(), 'state_dict')
        print('Took: ', str(datetime.timedelta(seconds=time() - epoch_start)))

    print('Finished in ', round(time() - start_t, 2), 's')
