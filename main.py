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
import Config

config = Config.get_config()
cifar_path = config["paths"]["cifar"]
model_path = config["paths"]["model"]
classes_path = config["paths"]["classes"]
num_classes = config["train"]["num_classes"]
num_epochs = config["train"]["num_epochs"]
batch_size = config["loader"]["batch_size"]
num_workers = config["loader"]["num_workers"]
photo_dimension = config["photo"]["dimension"]
def prompt_for_string(title: str, prompt: str) -> str:
    root = tk.Tk()
    root.withdraw()
    path = simpledialog.askstring(title=title, prompt=prompt)
    root.destroy()
    return path

def initialize_net() -> Net:
    net = NetHandler.initialize_net(3, photo_dimension, num_classes)
    NetHandler.save(net, model_path)
    print("Model successfully initialized")
    print("Model saved under path: " + model_path)
    return net

def load_model() -> (Net, list[str]):
    default_choice = "Default"
    path_choice = "Path"
    choice = Gui.option_dialog("Source", "Select source of model:", [default_choice, path_choice])

    user_model_path = None
    user_categories_path = None
    if choice is None:
        return
    elif choice == default_choice:
        user_model_path = model_path
        user_categories_path = classes_path
    elif choice == path_choice:
        is_path = False
        user_model_path = None
        while not is_path:
            user_model_path = prompt_for_string(title="Model path", prompt="Provide path to model:")
            if user_model_path is None:
                print("Loading cancelled by user...")
                return None
            user_model_path = Path(user_model_path)
            is_path = user_model_path.is_dir()

        is_path = False
        user_categories_path = None
        while not is_path:
            user_categories_path = prompt_for_string(title="Classes categories", prompt="Provide path to categories:")
            if user_categories_path is None:
                print("Loading cancelled by user...")
                return None
            user_categories_path = Path(user_categories_path)
            is_path = user_categories_path.is_dir()
    net = NetHandler.load(user_model_path, 3, photo_dimension, num_classes)
    categories = NetHandler.load_classes(user_categories_path)
    print("Model successfully loaded from: " + str(user_model_path))
    print("Categories handled: " + str(categories))
    return net, categories

def save_model(net: Net, classes: list[str], path = classes_path):
    if net is None:
        print("Model can't be saved: It is null")
        return
    NetHandler.save(net, model_path)
    print("Model saved under path: " + model_path)
    if classes is None:
        print("Classes can't be saved: They are null")
        return
    with open(classes_path, 'w') as f:
        for cls in classes:
            f.write(f"{cls}\n")
def enter_train_mode(net: Net) -> list[str]:
    train_set = torchvision.datasets.CIFAR10(root=cifar_path, train=True,
                                             download=True, transform=resize_tensor_normalize)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              # A wrapper that loads data from trainset in mini-batches using parallelism
                                              shuffle=True, num_workers=num_workers)

    net.train_on(train_loader, num_epochs)

    print("Model finished training")
    return train_set.classes

def enter_test_mode(net: Net, model_classes: list[str]):
    cifar_choice = "CIFAR-100"
    directory_choice = "Directory"
    choice = Gui.option_dialog("Source", "Select source of images:", [cifar_choice, directory_choice])

    data_loader = None
    test_classes = None
    if choice is None:
        return
    elif choice == cifar_choice:
        test_set = torchvision.datasets.CIFAR10(root=cifar_path, train=False,
                                                download=True, transform=resize_tensor_normalize)
        data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)
        test_classes = test_set.classes

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
        test_classes = data_set.classes
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    else:
        print("Unknown behavior")
        return

    print(model_classes)
    print(test_classes)

    net.test(data_loader, model_classes, test_classes)
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
    categories = None
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
            net, categories = load_model()
        elif choice == save_model_choice:
            save_model(net, categories)
        elif choice == train_choice:
            categories = enter_train_mode(net)
        elif choice == test_choice:
            enter_test_mode(net, categories)
        elif choice == categorize_choice:
            enter_categorize_mode()


        else:
            print("Unknown behavior")
            break

if __name__ == '__main__':
    main()