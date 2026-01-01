import torch 
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from torchvision import datasets as data
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

device = torch.device('cpu')

#Model
class CIFAR10A(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CIFAR10A, self).__init__() 
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 50)
        self.fc5 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)


model = CIFAR10A(1024, 10).to(device)

#Hyper-parameter
lr = 1e-4
epoch = 5

#Initialize optimizer and criterion
criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

#Training loader
train_dataset = data.CIFAR10(root='./data', train=True, download=False, transform=ToTensor())
test_dataset = data.CIFAR10(root='./data', train=False, download=False, transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

#Training loop
for batch_idx, (w_data ,target) in enumerate(train_loader):
    w_data = w_data.to(device)
    target = target.to(device)

    #Flatten the image
    w_data = w_data.reshape(w_data.shape[0], -1)

    #Forward pass
    scores = model(w_data)
    loss = criterion(scores,target)

    #Backward pass
    torch.zero_grad()
    loss.backward()
    optimizer.step()

#Accuracy function
def check_accuracy(model, test_loader):
    num_samples = 0
    num_correct = 0

    with torch.no_grad():
        for x, y in test_loader:
            x.to(device)
            x.to(device)
            x = x.reshape(x.shape[0], -1)
            scores = model(x)
            _, predictions = scores.max(1)
            if predictions.shape[0] == y.shape[0]:
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
            
    acc = float(num_correct)/float(num_samples)
    print(f'Accuracy: {acc*100:.2f}%')

    model.train()

check_accuracy(model, test_loader)


