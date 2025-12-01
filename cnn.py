#Importing the modules from torch libraries and also loading the data 
import torch #importing the main module
import torch.nn as nn #import the neural network module from torch
import torch.optim as optim #importing the optimizer algorithims
import torch.nn.functional as F #importing the functional key 
from torch.utils.data import DataLoader #importing the data initializer
import torchvision.datasets as datasets #Importing the data
import torchvision.transforms as transforms#Importing the data transformer


class CNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels= 16, kernel_size=(3,3), stride=(1,1), padding=(1,1)) 
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x
# Setup the device
device = torch.device("cpu")
#Hyperparameters
batch = 64
learning_rate = 0.01
epoch = 5


#Training and testing data
#We are importing and then converting the dataset into a tensor by torchvision.transforms and then loading it into a DataLoader
train_dataset =datasets.MNIST(root = "dataset/", train=True, transform=transforms.ToTensor(), download = False)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)
test_dataset = datasets.MNIST(root = "dataset/", train=False, transform=transforms.ToTensor(), download = False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch, shuffle=True)

#Setting up the device and sending it to cpu
model = CNN().to(device)

#Loss and Optmizer
# A loss function is to measure how bad or good a models prediction are to the actual results
#It gives us a single number probably decimals
#Entropy Loss functions are used for classification problems such as Cross, BinaryCross, Sparse, Categorical etc.
#A optimizer is used to change weights to minimize bad predictions 
#Adam is a optimizer that combines the power of RMSprop and AdaGrad so it is very suitable for almost every type of neural networks
#In other words it uses recent gradients and historic gradients too
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Train the network
for epoch in range(epoch):
    # over here we are taking the batch idx and taking the data and target from the train_loader 
    for batch_idx, (data, targets) in enumerate(train_loader):
        # putting the data and values in the device
        data = data.to(device)
        targets = targets.to(device)
        

        # forward the data into the model
        scores = model(data)
        loss = criterion(scores, targets)
        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        #optmizing the weights
        optimizer.step()

#checking accuracy
def check_accuracy(loader, model): #making the function to check the accuracy of the model
    if loader.dataset.train: #This tells the developer if the model is working training data or testing data
        print("Training data")
    else:
        print("Checking accuracy on testing data data")
    num_correct = 0  #This initializes num_correct
    num_samples = 0 #This initializes num_samples
    model.eval() #puts the model into evalution mode

    with torch.no_grad():
        for x,y in loader:  #In each loader taking out the image tensors and expected result an sending them into the device
            x = x.to(device)
            y = y.to(device)
            

            scores = model(x) #forwarding the data
            _, predictions = scores.max(1) #taking the predictions
            num_correct += (predictions == y).sum() # checking the if the prediction is correct
            num_samples += predictions.size(0) #adding it into the samples
        print(f"Got {num_correct}/ {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")

    model.train()

check_accuracy(test_loader, model)

def save_model(model):
    with open("cnn_model.pth", "wb") as f:
        torch.save(model.state_dict(), f)
        print("Model saved to model.pth")

#Saving the model state dict which contains all the weights and biases of the model into a file named model.pth