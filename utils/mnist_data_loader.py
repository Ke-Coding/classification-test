import torchvision as tv
from torch.utils.data import DataLoader

__all__ = ['MNIST_INPUT_SIZE', 'MNIST_INPUT_CHANNELS', 'MNIST_NUM_CLASSES', 'MNIST_MEAN', 'MNIST_STD',
           'get_train_loader', 'get_test_loader']


MNIST_INPUT_SIZE = 28
MNIST_INPUT_CHANNELS = 1
MNIST_NUM_CLASSES = 10
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def get_train_loader(batch_size):
    return DataLoader(
        tv.datasets.MNIST('mnist data', train=True, download=True,
                          transform=tv.transforms.Compose([
                              tv.transforms.ToTensor(),
                              tv.transforms.Normalize(
                                  (MNIST_MEAN,), (MNIST_STD,)
                              )
                          ])),
        batch_size=batch_size,
        shuffle=True)


def get_test_loader(batch_size):
    return DataLoader(
        tv.datasets.MNIST('mnist data', train=False, download=True,
                          transform=tv.transforms.Compose([
                              tv.transforms.ToTensor(),
                              tv.transforms.Normalize(
                                  (MNIST_MEAN,), (MNIST_STD,)
                              )
                          ])),
        batch_size=batch_size,
        shuffle=False)
