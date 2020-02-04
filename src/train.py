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

# batch size gets multiplied by 3 later
BATCH_SIZE = 25
# Number of total batches trained on
RUNS = 10000 // BATCH_SIZE
RUN_NAME = 'n_10k_b25_dmpi_br0.75_lr1e-5'


def train():
    start_t = time()
    print('\nTraining')

    writer = SummaryWriter('runs_new/' + RUN_NAME)

    # Load data
    datasets = load_dataset('all')
    s_train = datasets['train']
    s_db = datasets['db']

    # Load NN
    net = Net().double()
    optimizer = optim.Adam(net.parameters(), lr=1e-5)

    loss_sum = 0
    for run in range(1, RUNS + 1):
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
            test(run * BATCH_SIZE, s_test=datasets['test'], s_db=datasets['db'], writer=writer)

        loss.backward()
        optimizer.step()

    print('Finished in ', str(datetime.timedelta(seconds=time() - start_t)), 's\n')


if __name__ == '__main__':  # Only execute if called
    train()
