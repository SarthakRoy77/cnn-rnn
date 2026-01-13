import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.optim import Adam

class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(64 * 1 * 1, 256)
        self.fc1 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = self.fc1(x)

        return x

device = torch.device("cpu")

lr = 0.0001
epochs = 10
cnn_model = CNN1().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(cnn_model.parameters(), lr=lr)

train_dataset = datasets.MNIST(root="/dataset", train=True, download=True
                               , transform=ToTensor())
test_dataset = datasets.MNIST(root="/dataset", train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for epoch in range(epochs):
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)

        scores = cnn_model(data)
        loss = criterion(scores, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def check_accuracy(model, loader):
    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for tsdata, tstarget in loader:
            tsdata = tsdata.to(device)
            tstarget = tstarget.to(device)

            tsscores = model(data)
            _, preds = tsscores.max(1)

            num_correct += (preds == tstarget).sum().item()
            num_samples += preds.size(0)

    acc = 100 * num_correct / num_samples
    model.train()
    print(f"The accuracy is : {acc:.2f}%")

check_accuracy(cnn_model, test_loader)
