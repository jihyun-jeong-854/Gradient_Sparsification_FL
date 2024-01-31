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
        test_data_dir = os.path.join(DATA_PATH, dataset, 'test/')

        test_file =  os.path.join(test_data_dir, str(idx) + '.npz')
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, is_train=True):

    X_ , y_ = [], []
    for id in idx:
        data = read_data(dataset, id, is_train)
        X_.extend(torch.Tensor(data['x']).type(torch.float32))
        y_.extend(torch.Tensor(data['y']).type(torch.int64))
    # return X_train, y_train
    data_ = [(x, y) for x, y in zip(X_, y_)]
    return data_



