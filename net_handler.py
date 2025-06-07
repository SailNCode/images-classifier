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
def confusion_matrix_to_list(confusion_matrix):
    if not confusion_matrix:
        return []

    labels = sorted(confusion_matrix.keys())

    rows = [["True\\Pred"] + labels]

    for true_label in labels:
        row = [true_label]
        for pred_label in labels:
            count = confusion_matrix[true_label].get(pred_label, 0)
            row.append(count)
        rows.append(row)

    return rows

