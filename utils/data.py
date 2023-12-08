import numpy as np
import os
import torch

DATA_PATH = os.path.join(os.path.abspath('.'),'dataset')

def read_data(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join(DATA_PATH, dataset, 'train/')

        train_file = os.path.join(train_data_dir, str(idx) + '.npz')
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join(DATA_PATH, dataset, 'test')

        test_file =  os.path.join(test_data_dir, str(idx) + '.npz')
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, is_train=True):
   
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)
        # return X_train, y_train
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        # return X_test, y_test
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


