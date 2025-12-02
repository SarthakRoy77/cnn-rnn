#Importing the modules from torch libraries and also loading the data 
import torch #importing the main module
import torch.nn as nn #import the neural network module from torch
import torch.optim as optim #importing the optimizer algorithims
import torch.nn.functional as F #importing the functional key 
from torch.utils.data import DataLoader #importing the data initializer
import torchvision.datasets as datasets #Importing the data
import torchvision.transforms as transforms#Importing the data transformer

device = torch.device("cpu")
#Hyperparameters
batch = 64
num_classes = 10
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
learning_rate = 0.02
epoch = 2

#Create a RNN 
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, h0)
        out.reshape(out.shape[0], -1)
        out = self.fc(out)

        return out 

#Training and testing data
#We are importing and then converting the dataset into a tensor by torchvision.transforms and then loading it into a DataLoader
train_dataset =datasets.MNIST(root = "dataset/", train="True", transform=transforms.ToTensor(), download = False)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)
test_dataset = datasets.MNIST(root = "dataset/", train="False", transform=transforms.ToTensor(), download = False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch, shuffle=True)

#Setting up the device and sending it to cpu
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

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
        data = data.to(device).squeeze(1)
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
        print("Checking accuracy on the testing data.. ")
    else:
        print("Checking accuracy on training data")
    num_correct = 0  #This initializes num_correct
    num_samples = 0 #This initializes num_samples
    model.eval() #puts the model into evalution mode

    with torch.no_grad():
        for x,y in loader:  #In each loader taking out the image tensors and expected result an sending them into the device
            x = x.to(device).squeeze(1)
            y = y.to(device)

            scores = model(x) #forwarding the data
            _, predictions = scores.max(1) #taking the predictions
            num_correct += (predictions == y).sum() # checking the if the prediction is correct
            num_samples += predictions.size(0) #adding it into the samples
        print(f"Got {num_correct}/ {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")

    model.train()

check_accuracy(test_loader, model)

def save_model(model):
    with open("model.pth", "wb") as f:
        torch.save(model.state_dict(), f)
        print("Model saved to model.pth")

#Saving the model state dict which contains all the weights and biases of the model into a file named model.pth