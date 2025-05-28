import torch.nn as nn
import torch.nn.functional as F
import sys
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from collections import defaultdict
from tqdm import tqdm  # if not installed: pip install tqdm

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
    from tqdm import tqdm  # if not installed: pip install tqdm

    def test(self, test_loader: DataLoader, classes: list[str]):
        self.eval()

        correct_total = 0
        total_samples = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)

                total_samples += labels.size(0)
                correct_total += (predicted == labels).sum().item()

                for label, prediction in zip(labels, predicted):
                    class_total[classes[label]] += 1
                    if prediction == label:
                        class_correct[classes[label]] += 1

        print(f'\nTotal accuracy: {100 * correct_total / total_samples:.2f}%')
        print("\nPer-class accuracy:")
        for class_name in classes:
            if class_total[class_name] > 0:
                acc = 100 * class_correct[class_name] / class_total[class_name]
                print(f'{class_name:15s}: {acc:5.2f}%')
        sys.stdout.flush()
