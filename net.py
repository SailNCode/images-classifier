import torch.nn as nn
import torch.nn.functional as F
import sys
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from collections import defaultdict
from tqdm import tqdm
from shutil import copy2
from pathlib import Path
from PIL import Image

import Gui
import ImageHandler


class Net(nn.Module):
    def __init__(self, in_channels: int, dim: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5) #produces 6 filters res size: 32-5+1= 28
        self.pool = nn.MaxPool2d(2, 2) #stride - moving by 2
        self.conv2 = nn.Conv2d(6, 16, 5)
        #for config purposes:
        dummy_input = torch.zeros(1, in_channels, dim, dim)
        dummy = self.pool(F.relu(self.conv1(dummy_input)))
        dummy = self.pool(F.relu(self.conv2(dummy)))
        num_features = dummy.numel()

        self.fc1 = nn.Linear(num_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_on(self, train_loader: DataLoader, num_epochs: int):
        if train_loader is None:
            print("No training data. Aborting...")
            return

        print("Training initiated...")
        num_batches = len(train_loader)
        print(f"Images to process: {num_batches * train_loader.batch_size} (in {num_batches} batches)")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(num_epochs):

            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                optimizer.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 250 == 249:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 250:.3f}')
                    sys.stdout.flush()
                    running_loss = 0.0

        print('Finished Training')
        sys.stdout.flush()

    def test(self, test_loader: DataLoader, model_classes: list[str], test_classes: list[str]):
        if test_loader is None:
            print("No testing data. Aborting...")
            return
        self.eval()

        correct_total = 0
        total_samples = len(test_loader.dataset)
        print(f"Images to test: {total_samples}")
        class_correct = defaultdict(int)
        class_total = defaultdict(int)


        confusion_matrix = {true_cls: {pred_cls: 0 for pred_cls in model_classes} for true_cls in model_classes}

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)

                for label, prediction in zip(labels, predicted):
                    label_idx = label.item()
                    prediction_idx = prediction.item()

                    label_name = test_classes[label_idx]
                    prediction_name = model_classes[prediction_idx]

                    class_total[label_name] += 1
                    confusion_matrix[label_name][prediction_name] += 1

                    if label_name == prediction_name:
                        class_correct[label_name] += 1
                        correct_total += 1

        print(f'\nTotal accuracy: {100 * correct_total / total_samples:.2f}%')

        print("\nPer-class accuracy:")
        for class_name in model_classes:
            if class_total[class_name] > 0:
                acc = 100 * class_correct[class_name] / class_total[class_name]
                print(f'{class_name:15s}: {acc:5.2f}%')

        Gui.display_confusion_matrix(confusion_matrix)

    def categorize_and_save_images(
            self,
            input_dir: str,
            output_dir: str,
            model_classes: list[str],
            supported_exts=(".jpg", ".jpeg", ".png")
    ):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        self.eval()

        image_paths = list(input_dir.rglob("*"))
        num_all_files = len(image_paths)
        image_paths = [p for p in image_paths if p.suffix.lower() in supported_exts]
        num_images = len(image_paths)
        print(f"Found {num_images} images in {num_all_files - num_images + 1} folders.")


        for img_path in tqdm(image_paths, desc="Categorizing images"):
            try:
                with Image.open(img_path) as img:
                    img_tensor = ImageHandler.resize_tensor_normalize(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = self(img_tensor)
                        _, pred_idx = torch.max(output, 1)
                        pred_class = model_classes[pred_idx.item()]
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

            dest_dir = output_dir / pred_class
            dest_dir.mkdir(parents=True, exist_ok=True)
            copy2(img_path, dest_dir / img_path.name)

        print("Categorizing completed.")
        print("Images saved under: " + output_dir.__str__())
        Gui.show_category_browser(output_dir)