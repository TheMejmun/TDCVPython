from time import time
import torch
from data import load_dataset
from batch_gen import generate_triplet_batch
from nn import Net
from loss import *
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import datetime
from test import test

RUNS = 10000
# batch size gets multiplied by 3 later
BATCH_SIZE = 25


def train():
    start_t = time()
    print('\nTraining')

    writer = SummaryWriter('runs/train')

    # Load data
    datasets = load_dataset('all')
    s_train = datasets['train']
    s_db = datasets['db']

    # Load NN
    net = Net().double()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    for run in range(RUNS):
        torch.save(net.state_dict(), f='state_dict.pth')

        batch = generate_triplet_batch(s_train, s_db, BATCH_SIZE)
        results = list()

        optimizer.zero_grad()

        for i in batch:
            results.append(net(i[0].view(1, 3, 64, 64)))

        loss = l_triplets(results) + l_pairs(results)

        if run % 10 == 0:
            print('Run: ', run, '\tLoss: ', float(loss))
            writer.add_scalar(tag='scaled_training_loss',
                              scalar_value=float(loss) / BATCH_SIZE,
                              global_step=run)

        if run % 1000 == 0:
            writer.close()
            test(run, s_test=datasets['test'], s_db=datasets['db'])
            writer = SummaryWriter('runs/train')

        loss.backward()
        optimizer.step()

    writer.close()
    print('Finished in ',  str(datetime.timedelta(seconds=round(time() - start_t, 2))), 's\n')


if __name__ == '__main__':  # Only execute if called
    train()
