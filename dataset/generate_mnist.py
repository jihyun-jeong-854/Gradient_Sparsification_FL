import numpy as np
import os
import sys
import random
import argparse
import torch
import torchvision
import torchvision.transforms as transforms

import os
from dataset_utils import check, separate_data, split_data, save_file

DATA_PATH = os.path.dirname(os.path.abspath(__file__)) 
# Allocate data to users
def generate_mnist(data_path, save_path,  num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # Setup directory for train/test data
    config_path = os.path.join(save_path, 'config.json')
    train_path = os.path.join(save_path, 'train')
    test_path = os.path.join(save_path, 'test')
    
    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    # Get MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.MNIST(
        root=os.path.join(data_path,"rawdata"), train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=os.path.join(data_path,"rawdata"), train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.repeat(1,3,1,1).cpu().detach().numpy())
    dataset_image.extend(testset.data.repeat(1,3,1,1).cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    print(len(dataset_image), len(dataset_label))
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--niid', type=bool, default=False)
    parser.add_argument('--balance', type=bool, default=True)
    parser.add_argument('--partition', type=bool, default=None)
    parser.add_argument('--num_clients', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--save_path', type=str, default='mnist')
    args = parser.parse_args()
    
    random.seed(1)
    np.random.seed(1)
    num_classes = 10
   
    data_path = os.path.join(DATA_PATH, args.dataset)
    save_path = os.path.join(DATA_PATH, args.save_path)
    
    generate_mnist(data_path, save_path, args.num_clients, num_classes, args.niid, args.balance, args.partition)