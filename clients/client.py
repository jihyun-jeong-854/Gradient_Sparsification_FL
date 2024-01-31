import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data import read_client_data
from operator import itemgetter 

from models.mobilenet_v2_femnist import MobileNetV2FEMNIST
class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, **kwargs):
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.train_samples = train_samples # 합칠 npz 파일 개수
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.learning_rate_decay = args.learning_rate_decay
        self.local_epochs = args.local_epochs
        self.topk = args.topk
        self.topk_algo = args.topk_algo
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        
        self.trainloader = self.load_train_data()
        self.testloader = self.load_test_data()
        self.iter_trainloader = iter(self.trainloader)
        
       
        
    def load_train_data(self):
        data_idx = [id for id in range(self.id*self.train_samples, (self.id+1)*self.train_samples)]
        train_data = read_client_data(self.dataset, data_idx, is_train=True)
        print('client id: ', self.id ,'train data: ', len(train_data) )
        self.train_data_size = len(train_data)
        return DataLoader(train_data, self.batch_size, drop_last=True, shuffle=True)
       
  

    def load_test_data(self):
        data_idx = [id for id in range(self.id*self.train_samples, (self.id+1)*self.train_samples)]
        test_data = read_client_data(self.dataset, data_idx, is_train=False)
        return DataLoader(test_data,len(test_data), drop_last=True, shuffle=True)
    
    def get_next_train_batch(self,local_rounds):
        xs = []
        ys = []
        for round in range(local_rounds):
            try:
                # Samples a new batch for persionalizing
                (x,y) = next(self.iter_trainloader)
                
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                self.iter_trainloader = iter(self.trainloader)
                (x, y) = next(self.iter_trainloader)
                
            xs.extend(x)
            ys.extend(y)
        xs = torch.stack(xs)
        ys = torch.stack(ys)
        return xs, ys

        
    # def get_next_batch(self,is_shuffled=False):
    #     batch_size = self.batch_size
    #     assert batch_size<=self.train_samples, \
    #         "Batch size %d is larger than n_data %d"%(batch_size,self.train_samples)
    #     start_idx = self.cur_idx
    #     end_idx = self.cur_idx + batch_size
    #     if end_idx > self.train_samples:
    #         indices = list(range(start_idx, self.train_samples)) + \
    #             list(range(0, batch_size - (self.train_samples-start_idx)))
    #         self.cur_idx = batch_size - (self.train_samples-start_idx)
    #     else:
    #         indices = list(range(start_idx, end_idx))
    #         self.cur_idx = end_idx
    #     assert(len(indices) == end_idx - start_idx)
        
    #     return self.train_x[indices[i] for i in indices], self.train_y[indices[i] for i in indices]
    
    def set_parameters(self, model):
        self.model.load_state_dict(model.state_dict())
        # for new_param, old_param in zip(model.parameters(), self.model.parameters()):
        #     old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):

        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in self.testloader:
        
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
          
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
          
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))