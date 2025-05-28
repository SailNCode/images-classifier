from net import Net
import torch

def initialize_net(n_channels: int, photo_dim: int, num_categories: int) -> Net:
    return Net(n_channels, photo_dim, num_categories)

def save(net: Net, path: str):
    torch.save(net.state_dict(), path)

def load(path: str, in_channels: int, dim: int, num_classes: int) -> Net:
    net = Net(in_channels, dim, num_classes)
    net.load_state_dict(torch.load(path, weights_only=True))
    return net
def load_classes(path: str) -> list[str]:
    with open(path, "r") as file:
        categories = [line.strip() for line in file]

    return categories


