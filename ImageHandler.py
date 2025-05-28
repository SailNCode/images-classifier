import torchvision.transforms as transforms
import random
from collections import defaultdict
import Config

config = Config.get_config()
photo_dimension = config["photo"]["dimension"]

resize_tensor_normalize = transforms.Compose(
        [transforms.Resize((photo_dimension, photo_dimension)),
         transforms.ToTensor(), #splits into separate list of red, green and blue channels and divides each value by 255, so x e [0,1]
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalizes to [-1,1] by (x-mean)/std
