import torch.nn as nn
import torch.nn.functional as F
import sys
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from collections import defaultdict
from tqdm import tqdm
import pandas as pd



def display_confusion_matrix(class_correct: defaultdict, class_total: defaultdict):
    classes = sorted(class_total.keys())

    # Initialize confusion matrix with zeros
    matrix = {true: {pred: 0 for pred in classes} for true in classes}

    for cls in classes:
        correct = class_correct.get(cls, 0)
        total = class_total.get(cls, 0)
        incorrect = total - correct

        # Place correct predictions on the diagonal
        matrix[cls][cls] = correct

        # Distribute incorrects (we’ll just assign to the other class for illustration)
        other_classes = [c for c in classes if c != cls]
        if other_classes:
            matrix[cls][other_classes[0]] = incorrect

    # Convert to DataFrame for better display
    df = pd.DataFrame(matrix).T  # rows: true labels, cols: predicted labels
    print(df)


class Net(nn.Module):
    def __init__(self, in_channels: int, dim: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5) #produces 6 filters res size: 32-5+1= 28
        self.pool = nn.MaxPool2d(2, 2) #stride - moving by 2
        self.conv2 = nn.Conv2d(6, 16, 5)
        #for config purposes:
        dummy_input = torch.zeros(1, in_channels, dim, dim)  # batch=1, channels=3, 32x32 image
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
        print("Training initiated...")
        num_batches = len(train_loader)
        print(f"Images to process: {num_batches * train_loader.batch_size} (in {num_batches} batches)")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        #optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(num_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 250 == 249:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    sys.stdout.flush()
                    running_loss = 0.0

        print('Finished Training')
        sys.stdout.flush()

    def test(self, test_loader: DataLoader, model_classes: list[str], test_classes: list[str]):
        self.eval()

        correct_total = 0
        total_samples = len(test_loader.dataset)
        print(f"total samples: {total_samples}")
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        true_labels = []
        predictions = []

        # Initialize confusion matrix as nested dict of ints
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

                    true_labels.append(label_name)
                    predictions.append(prediction_name)

                    class_total[label_name] += 1
                    confusion_matrix[label_name][prediction_name] += 1  # Update confusion matrix

                    if label_name == prediction_name:
                        class_correct[label_name] += 1
                        correct_total += 1

        print('True values: ', ' '.join(true_labels))
        print('Predictions: ', ' '.join(predictions))
        print(f'\nTotal accuracy: {100 * correct_total / total_samples:.2f}%')

        print("\nPer-class accuracy:")
        for class_name in model_classes:
            if class_total[class_name] > 0:
                acc = 100 * class_correct[class_name] / class_total[class_name]
                print(f'{class_name:15s}: {acc:5.2f}%')

        # Display confusion matrix as pandas DataFrame
        df_cm = pd.DataFrame(confusion_matrix).T
        print("\nConfusion Matrix:")
        pd.set_option('display.max_columns', None)
        print(df_cm)

        sys.stdout.flush()
