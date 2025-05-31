import torch
import torchvision
from net import Net
from ImageHandler import resize_tensor_normalize
from NetHandler import initialize_net, load, load_classes, save
import Gui
from torchvision import datasets
from torch.utils.data import DataLoader
from pathlib import Path
import Config


config = Config.get_config()
cifar_path = config["paths"]["cifar"]
model_path = config["paths"]["model"]
categories_path = config["paths"]["categories"]
num_classes = config["train"]["num_classes"]
num_epochs = config["train"]["num_epochs"]
batch_size = config["loader"]["batch_size"]
num_workers = config["loader"]["num_workers"]
photo_dimension = config["photo"]["dimension"]

def initialize_net() -> Net:
    net = initialize_net(3, photo_dimension, num_classes)
    print("Model successfully initialized")
    print("Model is kept in memory. Remember to save it!")
    return net

def load_model() -> (Net, list[str], str):
    choice = Gui.option_dialog("Source", "Select source of model:", ["Default", "Select"])

    user_model_path = None
    user_categories_path = None
    if choice is None:
        return
    elif choice == 1: #Default model
        if not model_path or not Path(model_path).is_file():
            print("Model is absent. Aborting...")
            return None, None, None
        if not categories_path or not Path(categories_path).is_file():
            print(categories_path)
            print("Categories are absent. Aborting...")
            return None, None, None
        user_model_path = model_path
        user_categories_path = categories_path
    elif choice == 2: #User choice
        print("Select model file")
        user_model_path = Gui.select_model_file("Select model file", [("PyTorch model", "*.pth"),])
        if user_model_path is None:
            print("Loading cancelled by user...")
            return None, None, None

        print("Select categories file")
        user_categories_path = Gui.select_model_file("Select categories file", [("Categories file", "*.txt"),])
        if user_categories_path is None:
            print("Loading cancelled by user...")
            return None, None, None
    try:
        net = load(user_model_path, 3, photo_dimension, num_classes)
        print("Model successfully loaded from: " + str(user_model_path))
    except:
        print("Invalid model. Aborting...")
        return None, None, None

    categories = load_classes(user_categories_path)
    print("Categories successfully loaded from " + str(user_categories_path) if categories else "No categories in file.")
    return net, categories, user_model_path

def save_model(net: Net, classes: list[str]) -> str:
    if net is None:
        print("Model can't be saved: It is null")
        return None
    save(net, model_path)
    print("Model saved under path: " + model_path)
    if classes is None or len(classes) == 0:
        print("Classes can't be saved: They are absent")
        return None
    with open(categories_path, 'w') as f:
        for cls in classes:
            f.write(f"{cls}\n")
    print("Classes saved under path: " + categories_path)
    return model_path
def enter_train_mode(net: Net) -> list[str]:
    if net is None:
        print("Model can't be trained: It is not loaded")
        return None
    train_set = torchvision.datasets.CIFAR10(root=cifar_path, train=True,
                                             download=True, transform=resize_tensor_normalize)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    net.train_on(train_loader, num_epochs)

    return train_set.classes

def enter_test_mode(net: Net, model_classes: list[str]):
    choice = Gui.option_dialog("Source", "Select source of images:", ["CIFAR-10", "Directory"])
    if net is None:
        print("Model can't be tested: It is not loaded")
        return

    data_loader = None
    test_classes = None
    if choice is None:
        return
    elif choice == 1:
        test_set = torchvision.datasets.CIFAR10(root=cifar_path, train=False,
                                                download=True, transform=resize_tensor_normalize)
        data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)
        test_classes = test_set.classes

    elif choice == 2:
        print("Select images directory...")
        path = Gui.select_directory("Select images directory")
        if path is None:
            print("Testing cancelled by user...")
            return

        data_set = datasets.ImageFolder(root=path, transform=resize_tensor_normalize)
        test_classes = data_set.classes
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    else:
        print("Unknown behavior")
        return

    net.test(data_loader, model_classes, test_classes)
def enter_categorize_mode(net: Net, categories):
    print("Select images directory...")
    src_path = Gui.select_directory("Select images directory")
    if src_path is None:
        print("Classification cancelled by user...")
        return

    print("Select where to save...")
    dest_path = Gui.select_directory("Select destination directory")
    if dest_path is None:
        print("Classification cancelled by user...")
        return


    net.categorize_and_save_images(
        src_path,
        dest_path,
        categories,
    )

def display_images():
    print("Select images directory...")
    src_path = Gui.select_directory("Select images directory")
    if src_path is None:
        print("Displaying cancelled by user...")
        return
    Gui.show_category_browser(Path(src_path))

def main():
    options = ("Initialize model", "Load model", "Save model", "Train", "Test","Categorize", "Display images")
    path = None
    net = None
    categories = None
    print("Welcome to the image classification tool!")

    while True:
        states = ("normal",
                  "normal",
                  "normal" if net and categories else "disabled",
                  "normal" if net else "disabled",
                  "normal" if net and categories else "disabled",
                  "normal" if net and categories else "disabled",
                  "normal")
        choice = Gui.show_main_panel("Image classification model", "Choose the action", options,
                                     model_path=path,
                                     categories=categories,
                                     states = states)
        match choice:
            case 1:
                net = initialize_net()
            case 2:
                net, categories, path= load_model()
            case 3:
                path = save_model(net, categories)
            case 4:
                categories = enter_train_mode(net)
            case 5:
                enter_test_mode(net, categories)
            case 6:
                enter_categorize_mode(net, categories)
            case 7:
                display_images()
            case None:
                break
if __name__ == '__main__':
    main()