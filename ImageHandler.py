import torchvision.transforms as transforms
import random
from collections import defaultdict

resize_tensor_normalize = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(), #splits into separate list of red, green and blue channels and divides each value by 255, so x e [0,1]
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalizes to [-1,1] by (x-mean)/std

def get_one_random_per_class(dataset, num_classes=100):
    class_to_indices = defaultdict(list)

    # Group indices by class
    for idx, (_, label) in enumerate(dataset):
        class_to_indices[label].append(idx)

    # Randomly pick one image index per class
    selected_indices = [random.choice(class_to_indices[c]) for c in range(num_classes)]

    # Create a subset of the dataset
    from torch.utils.data import Subset
    return Subset(dataset, selected_indices)