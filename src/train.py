import datetime
from time import time

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from batch_gen import generate_triplet_batch
from data import load_dataset
from loss import *
from nn import Net
from test import test

# batch size gets multiplied by 3 later
BATCH_SIZE = 25
# Number of total batches trained on
RUNS = 1000000 // BATCH_SIZE
RUN_NAME = 'n_1m_b25_dmpi_br0.85_lr1e-4'


def train(run_start=1):
    start_t = time()
    print('\nTraining')

    writer = SummaryWriter('runs_1m/' + RUN_NAME)

    # Load data
    datasets = load_dataset('all')
    s_train = datasets['train']
    s_db = datasets['db']

    # Load NN
    net = Net().double()
    # Resume training if not started from 1
    if run_start != 1:
        net.load()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    loss_sum = 0
    for run in range(run_start, RUNS + 1):
        # Save state_dict
        net.store()

        batch = generate_triplet_batch(s_train, s_db, BATCH_SIZE)
        results = list()

        optimizer.zero_grad()

        for i in batch:
            results.append(net(i[0].view(1, 3, 64, 64)))

        loss = l_triplets(results, batch) + l_pairs(results)
        # loss = l_triplets(results) + l_pairs(results)
        loss_sum += float(loss)

        if (run * BATCH_SIZE) % 100 == 0:
            loss_sum = loss_sum / 100
            print('Run: ', run * BATCH_SIZE, '\tLoss Average: ', loss_sum)
            writer.add_scalar(tag='avg_training_loss',
                              scalar_value=loss_sum,
                              global_step=run * BATCH_SIZE)
            loss_sum = 0

        if (run * BATCH_SIZE) % 10000 == 0:
            test(run=run * BATCH_SIZE, s_test=datasets['test'], s_db=datasets['db'], writer=writer)

        loss.backward()
        optimizer.step()

    print('Finished in ', str(datetime.timedelta(seconds=time() - start_t)), 's\n')


if __name__ == '__main__':  # Only execute if called
    train()
