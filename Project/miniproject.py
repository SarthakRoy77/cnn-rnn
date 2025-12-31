from torchvision import datasets as data
from torchvision.transforms import ToTensor
train_dataset = data.CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
test_dataset = data.CIFAR10(root='./data', train=False, download=True, transform=ToTensor())


