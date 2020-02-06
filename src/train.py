import datetime
from time import time

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from batch_gen import generate_triplet_batch
from data import load_dataset
from loss import *
from nn import Net
from test import test
from triplet_dataset import TripletDataset
from torch.utils.data import DataLoader

# batch size gets multiplied by 3 later
BATCH_SIZE = 25
RUN_NAME = 'b25_dmpi_br0.85_lr1e-3_bn_xn_dl'
EPOCHS = 64


def train(epoch=0, dynamic_margin=True, run_name=RUN_NAME):
    start_t = time()
    print('\nTraining')

    writer = SummaryWriter('runs' + run_name)

    # Load data
    datasets = load_dataset('all')
    dataset = TripletDataset(s_train=datasets['train'], s_db=datasets['db'], pusher_ratio=0.85)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load NN
    net = Net().double()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    exec = 0
    for epoch in range(epoch, EPOCHS):

        for i_anchors, i_pullers, i_pushers, anchors, pullers, pushers in dataloader:

            # Save state_dict
            net.store()

            loss_sum = 0
            optimizer.zero_grad()

            # Calculate descriptors
            o_anchors, o_pullers, o_pushers = net(i_anchors), net(i_pullers), net(i_pushers)

            if dynamic_margin:
                loss = l_triplets(o_anchors, o_pullers, o_pushers, anchors, pushers) + l_pairs(o_anchors, o_pullers)
            else:
                loss = l_triplets(o_anchors, o_pullers, o_pushers) + l_pairs(o_anchors, o_pullers)
            loss_sum += float(loss)

            if exec % 10 == 0:
                loss_sum = loss_sum / (BATCH_SIZE * 10)
                print('Epoch: ', epoch, '\tIteration: ', exec, '\tLoss Average: ', loss_sum)
                writer.add_scalar(tag='avg_training_loss',
                                  scalar_value=loss_sum,
                                  global_step=exec)
                loss_sum = 0

            if exec % 1000 == 0:
                test(run=exec, s_test=datasets['test'], s_db=datasets['db'], writer=writer)

            loss.backward()
            optimizer.step()

            exec += 1

    print('Finished in ', str(datetime.timedelta(seconds=time() - start_t)), 's\n')


if __name__ == '__main__':  # Only execute if called
    train(dynamic_margin=True, run_name='b25_dmpi_br0.85_lr1e-3_bn_xn_dl')
    train(dynamic_margin=True, run_name='b25_sm0.01_br0.85_lr1e-3_bn_xn_dl')
