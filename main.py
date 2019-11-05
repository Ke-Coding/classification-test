import time
import torch as tc
from torch import nn
from torch.nn import functional
from torch.optim import SGD, Adam
from torch.optim.adagrad import Adagrad
from torch.optim.rmsprop import RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import *
from utils import *


USING_GPU = tc.cuda.is_available()
BASIC_LR = 0.0002
WEIGHT_DECAY = 0.00001
OP_MOMENTUM = 0.9
MIN_LR = 0.0000175
MAX_EPOCH = 4
BATCH_SIZE = 128


def test(net: nn.Module):
    test_loader = get_test_loader(BATCH_SIZE)
    test_set_size = len(test_loader)
    
    net.eval()
    test_acc, test_loss = 0., 0.
    for (x, y) in test_loader:
        global USING_GPU
        if USING_GPU:
            x, y = x.cuda(), y.cuda()
        y_hat = net(x)
        test_acc += y_hat.argmax(dim=1).eq(y).sum().item() / y_hat.size(0)
        test_loss += functional.cross_entropy(y_hat, y).item()
    test_acc /= test_set_size
    test_loss /= test_set_size
    net.train()
    
    return test_acc, test_loss


def train(net: nn.Module):
    net.train()
    train_loader = get_train_loader(batch_size=BATCH_SIZE)
    train_set_size = len(train_loader)
    
    train_acc_history, test_acc_history = [], []
    train_loss_history, test_loss_history = [], []
    lr_history = []
    
    train_record_freq = 16
    test_record_freq = train_record_freq * 2
    lr_record_freq = train_record_freq * 2
    print_freq = train_record_freq * 8
    
    optimizer = SGD(net.parameters(), lr=BASIC_LR, weight_decay=WEIGHT_DECAY, momentum=OP_MOMENTUM)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCH * train_set_size, eta_min=MIN_LR)
    print('hyper params: epoch:%d, batch size:%d, coslr:%g => %g for %d epoch, weight decay:%g, momentum:%g\n' %
          (MAX_EPOCH, BATCH_SIZE, BASIC_LR, MIN_LR, MAX_EPOCH, WEIGHT_DECAY, OP_MOMENTUM)
    )
    
    net.train()
    totol_batch_idx = 0  # [0, MAX_EPOCH * train_set_size)
    for epoch in range(MAX_EPOCH):
        for batch_idx, (x, y) in enumerate(train_loader):
            global USING_GPU
            if USING_GPU:
                x, y = x.cuda(), y.cuda()
            # x: (batch_size, 1, 28, 28), y: (batch_size)
            y_hat = net(x)  # y_hat: (batch_size, 10), y_hat.argmax(dim=1): (batch_size)
            loss = functional.cross_entropy(y_hat, y)
            if totol_batch_idx % train_record_freq == train_record_freq - 1:
                train_acc = y_hat.argmax(dim=1).eq(y).sum().item() / BATCH_SIZE
                train_acc_history.append((totol_batch_idx, train_acc))
                train_loss = loss.item()
                train_loss_history.append((totol_batch_idx, train_loss))
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # learning rate decay
            scheduler.step(totol_batch_idx)
            
            if totol_batch_idx % lr_record_freq == lr_record_freq - 1:
                lr_history.append((totol_batch_idx, scheduler.get_lr()[0]))
            
            if totol_batch_idx % test_record_freq == test_record_freq - 1:
                test_acc, test_loss = test(net)
                test_acc_history.append((totol_batch_idx, test_acc))
                test_loss_history.append((totol_batch_idx, test_loss))
            if totol_batch_idx % print_freq == print_freq - 1:
                print('[%4.2f%%] epoch: %2d  |  batch_idx: %3d  |'
                      '  tr-acc: %4.2f%%  |  te-acc: %4.2f%%  |'
                      '  tr-loss: %7.4f  |  te-loss: %7.4f  |'
                      '  lr: %.6f' % (
                          100 * (totol_batch_idx) / (MAX_EPOCH * train_set_size), epoch, batch_idx,
                          100 * train_acc_history[-1][1], 100 * test_acc_history[-1][1],
                          train_loss_history[-1][1], test_loss_history[-1][1],
                          lr_history[-1][1]
                      )
                )
            totol_batch_idx += 1
    return train_acc_history, test_acc_history, train_loss_history, test_loss_history, lr_history


def train_from_scratch(net):
    print('=== start training from scratch ===\n')
    train_acc_history, test_acc_history, train_loss_history, test_loss_history, lr_history = train(net)
    final_test_acc, _ = test(net)
    final_acc_str = '%.2f' % (100 * final_test_acc)
    print('\n=== final test acc: %s ===\n' % final_acc_str)
    net.save(final_acc_str)
    plot_curves(train_acc_history, test_acc_history, train_loss_history, test_loss_history, lr_history)


def fine_tune(net):
    global MAX_EPOCH, BASIC_LR
    MAX_EPOCH = 1
    BASIC_LR /= 5
    net.load('97.26')
    print('=== start fine tuning ===\n')
    train_acc_history, test_acc_history, train_loss_history, test_loss_history, lr_history = train(net)
    final_test_acc, _ = test(net)
    final_acc_str = '%.2f' % (100 * final_test_acc)
    print('\n=== final test acc: %s ===\n' % final_acc_str)
    net.save(final_acc_str)
    plot_curves(train_acc_history, test_acc_history, train_loss_history, test_loss_history, lr_history)


if __name__ == '__main__':
    # net: FCNet = FCNet(input_dim=MNIST_INPUT_SIZE * MNIST_INPUT_SIZE * MNIST_INPUT_CHANNELS,
    #                    output_dim=MNIST_NUM_CLASSES,
    #                    hid_dims=[128, 64],
    #                    dropout_p=None)
    
    print('\n=== cuda is available ===\n' if USING_GPU else '=== cuda is not available ===\n')
    
    BASIC_LR *= 25
    MIN_LR *= 25
    net: ConvNet = ConvNet(input_channels=1, num_classes=10,
                channels=[32, 48, 48, 64],
                strides=[1, 2, 1, 2],
                dropout_p=None)
    
    if USING_GPU:
        net = net.cuda()
    
    start_sec = time.time()
    # fine_tune(net)
    train_from_scratch(net)
    print('=== time cost: %.2fs ===\n' % (time.time() - start_sec, ))
