import torch
import torchvision
import NetHandler
from ImageHandler import resize_tensor_normalize
def main():

    batch_size = 4
    photo_dim = 32
    num_epochs = 10

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                             download=True, transform=resize_tensor_normalize)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, #A wrapper that loads data from trainset in mini-batches using parallelism
                                              shuffle=True, num_workers=2)

    classes = trainset.classes

    net = NetHandler.initialize_net(3, photo_dim, 100)

    net.train_on(trainloader, num_epochs)

    NetHandler.save(net, "./cifar_net.pth")
if __name__ == '__main__':
    main()