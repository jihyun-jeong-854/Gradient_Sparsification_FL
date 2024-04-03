import torch
import torch.nn as nn
import torch.nn.functional as func


class LeNetFEMNIST(nn.Module):
    def __init__(self,num_classes):
        super(LeNetFEMNIST,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(16,120,5)
        self.n1 = nn.Linear(120,84)
        self.relu = nn.ReLU()
        self.n2 = nn.Linear(84,num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = torch.flatten(x,1)
        x = self.n1(x)
        x = self.relu(x)
        x = self.n2(x)
        return x