import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

mat = np.zeros(shape=(10, 2))
for i in mat:
    i[0] = np.random.choice([0, 1, 2])
    if i[0] == 1:
        i[1] = np.random.choice([32, 64, 128, 256, 512])
mat = np.insert(mat, 0, [1, np.random.choice([32, 64, 128, 256, 512])], axis=0)
print(mat)
num_conv = np.count_nonzero(mat == 1)
idx = np.where(mat == 1)[0]


class DENN(nn.Module):
    def __init__(self):
        super(DENN, self).__init__()
        self.conv = torch.nn.Sequential()
        for row in range(len(mat)):
            if int(mat[row, 0]) == 2:
                self.conv.add_module("pool" + str(row), nn.MaxPool2d(2))
            elif int(mat[row, 0]) == 1:
                inc = 1
                for i in range(len(idx)):
                    outc = int(mat[idx[i], 1])
                    if i == len(idx)-1:
                        outc = 64
                    self.conv.add_module('conv' + str(i), nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1))
                    self.conv.add_module('relu' + str(i), nn.ReLU())
                    # self.conv.add_module("pool" + str(row), nn.MaxPool2d(2))
                    inc = outc
        self.fc = None

    def forward(self, x):
        x = self.conv(x)
        a = x.size(0)
        x = x.view(-1, a)
        if self.fc == None:
            self.fc = torch.nn.Sequential()
            self.fc.add_module("fc1", nn.Linear(a, 128))
            self.fc.add_module("fcrelu", nn.ReLU())
            self.fc.add_module("fc2", nn.Linear(128, 10))
        x = self.fc(x)
        return x

model = DENN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

num_epochs = 5
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
