import torch
import torchvision
import torchvision.transforms as transforms
import NetHandler

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(), #splits into separate list of red, green and blue channels and divides each value by 255, so x e [0,1]
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalizes to [-1,1] by (x-mean)/std

    batch_size = 4

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = testset.classes

    net = NetHandler.load("./cifar_net.pth", 3, 32, 100)

    dataiter = iter(testloader)
    next(dataiter)
    next(dataiter)
    next(dataiter)
    images, labels = next(dataiter)

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                  for j in range(4)))

if __name__ == '__main__':
    main()