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

RUNS = 1000
# batch size gets multiplied by 3 later
BATCH_SIZE = 25
RUN_NAME = 'r100b25dm2pi'


def train():
    start_t = time()
    print('\nTraining')

    writer = SummaryWriter('runs/' + RUN_NAME)

    # Load data
    datasets = load_dataset('all')
    s_train = datasets['train']
    s_db = datasets['db']

    # Load NN
    net = Net().double()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    loss_sum = 0
    for run in range(1, RUNS + 1):
        try:
            torch.save(net.state_dict(), f='state_dict.pth')
        except OSError:
            print('Wasn\'t able to save State dict')

        batch = generate_triplet_batch(s_train, s_db, BATCH_SIZE)
        results = list()

        optimizer.zero_grad()

        for i in batch:
            results.append(net(i[0].view(1, 3, 64, 64)))

        loss = l_triplets(results, batch) + l_pairs(results)
        loss_sum += float(loss)

        if run % 10 == 0:
            loss_sum = loss_sum / (BATCH_SIZE * 10)
            print('Run: ', run, '\tLoss Average: ', loss_sum)
            writer.add_scalar(tag='avg_training_loss',
                              scalar_value=loss_sum,
                              global_step=run)
            loss_sum = 0

        if run % 100 == 0:
            test(run, s_test=datasets['test'], s_db=datasets['db'], writer=writer)

        loss.backward()
        optimizer.step()

    print('Finished in ', str(datetime.timedelta(seconds=time() - start_t)), 's\n')


if __name__ == '__main__':  # Only execute if called
    train()
