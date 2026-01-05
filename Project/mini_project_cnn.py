import torch
import torch.nn as nn

class CNN1(nn.Module):
    def __init__(self, input_size, numclasses):
        super(CNN1, self).__init__()
        self.input_size = input_size
        self.num_classes = numclasses

    def forward(self, x):
        pass

device = torch.device("cpu")

#Hyperparameters
lr = 0.0001

cnn_model = CNN1(784,10).to(device)


