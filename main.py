import torch
import torchvision
from net import Net

import ImageHandler
from ImageHandler import resize_tensor_normalize
import NetHandler
import Gui
from torchvision import datasets
from torch.utils.data import DataLoader
import tkinter as tk
from tkinter import simpledialog
from pathlib import Path
#Config:
model_path = "./cifar_net.pth"
photo_dim = 32

num_classes = 100
batch_size = 20
num_workers = 2

#train:
num_epochs = 10

cifar_path = './data'

def prompt_for_string(title: str, prompt: str) -> str:
    root = tk.Tk()
    root.withdraw()
    path = simpledialog.askstring(title=title, prompt=prompt)
    root.destroy()

def initialize_net() -> Net:
    net = NetHandler.initialize_net(3, photo_dim, num_classes)
    NetHandler.save(net, model_path)
    print("Model successfully initialized")
    print("Model saved under path: " + model_path)
    return net

def load_model() -> Net:
    default_choice = "Default"
    path_choice = "Path"
    choice = Gui.option_dialog("Source", "Select source of model:", [default_choice, path_choice])

    path = None
    if choice is None:
        return
    elif choice == default_choice:
        path = model_path
    elif choice == path_choice:
        is_path = False
        path = None
        root = tk.Tk()
        root.withdraw()
        while not is_path:
            path = simpledialog.askstring(title="Input path to model", prompt="Provide path to collective directory:")
            if path is None:
                print("Loading cancelled by user...")
                return None
            path = Path(path)
            is_path = path.is_dir()
        root.destroy()
    net = NetHandler.load(path, 3, photo_dim, num_classes)
    print("Model successfully loaded from: " + path.__str__())
    return net

def save_model(net: Net):
    NetHandler.save(net, model_path)

def enter_train_mode(net: Net):
    train_set = torchvision.datasets.CIFAR100(root=cifar_path, train=True,
                                             download=True, transform=resize_tensor_normalize)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              # A wrapper that loads data from trainset in mini-batches using parallelism
                                              shuffle=True, num_workers=num_workers)

    net.train_on(train_loader, num_epochs)

    print("Model finished training")

def enter_test_mode(net: Net):
    cifar_choice = "CIFAR-100"
    directory_choice = "Directory"
    choice = Gui.option_dialog("Source", "Select source of images:", [cifar_choice, directory_choice])

    data_loader = None
    classes = None

    if choice is None:
        return
    elif choice == cifar_choice:
        test_set = torchvision.datasets.CIFAR100(root=cifar_path, train=False,
                                                download=True, transform=resize_tensor_normalize)
        data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)
        classes = test_set.classes

    elif choice == directory_choice:
        is_path = False
        path = None
        while (not is_path):
            path = prompt_for_string(title="Input directory", prompt="Provide path to collective directory:")
            print(path)
            if path is None:
                print("Testing cancelled by user...")
                return
            path = Path(path)
            is_path = path.is_dir()

        data_set = datasets.ImageFolder(root=path, transform=resize_tensor_normalize)
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        classes = data_set.classes
    else:
        print("Unknown behavior")
        return

    net.test(data_loader, classes)
    print("Finished testing")
def main():











    def enter_categorize_mode():
        pass

    initialize_net_choice = "Initialize model"
    load_model_choice = "Load model"
    save_model_choice = "Save model"
    train_choice = "Train"
    test_choice = "Test"
    categorize_choice = "Categorize"

    net = None
    while True:
        choice = Gui.option_dialog("Chose action", "Pick your favorite fruit:",
                                   [initialize_net_choice,
                                    load_model_choice,
                                    save_model_choice,
                                    train_choice,
                                    test_choice,
                                    categorize_choice])
        if choice is None:
            break
        elif choice == initialize_net_choice:
            net = initialize_net()
        elif choice == load_model_choice:
            net = load_model()
        elif choice == save_model_choice:
            save_model(net)
        elif choice == train_choice:
            enter_train_mode(net)
        elif choice == test_choice:
            enter_test_mode(net)
        elif choice == categorize_choice:
            enter_categorize_mode()


        else:
            print("Unknown behavior")
            break

if __name__ == '__main__':
    main()