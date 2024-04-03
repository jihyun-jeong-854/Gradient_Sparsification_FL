import torch
import os
import numpy as np
import h5py
import copy
import time

from sklearn.preprocessing import label_binarize
from sklearn import metrics

from utils.data import read_client_data
from torch.utils.data import DataLoader
from copy import deepcopy

class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.model_name = args.model_name
        self.global_model = copy.deepcopy(args.model)
        self.optimizer = torch.optim.SGD(self.global_model.parameters(), lr=self.learning_rate)
        self.algorithm = args.algorithm
       
        self.save_folder_name = args.save_folder_name

        self.clients = []
        self.selected_clients = []
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.client_ids = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
       
        self.topk = self.args.k1
        self.topk_algo = self.args.topk_algo
        self.num_writers = args.num_writers
        # self.test_loader = self.set_test_data()

    def set_clients(self, clientObj):
        self.client_ids = []
     
        total_samples = 0
        for i in range(self.num_clients) :
            client = clientObj(
                self.args,
                id=i,
                train_samples=self.num_writers,
            )
            train_data_size = client.train_data_size
            self.clients.append(client)
            self.client_ids.append(i)
            total_samples += train_data_size
   
            
    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients
    
    def set_test_data(self):
        test_data = []
        for i in range(self.num_clients):
            test_data.extend(read_client_data(self.dataset, i, is_train=False))
       
        return DataLoader(test_data, len(test_data), drop_last=False, shuffle=True) 

    def send_models(self):
        assert len(self.clients) > 0
    
        for client in self.clients:
            start_time = time.time()

            client.model.load_state_dict(self.global_model.state_dict())
            
            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)
            
    def save_global_model(self,cur_epoch):
        model_path = os.path.join("checkpoint")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # global_path = os.path.join(model_path, self.model_name + self.topk_algo + str(self.topk) + "_server" + ".pt")
        # client_path = os.path.join(model_path, self.model_name +self.topk_algo + str(self.topk)  + "_client_epoch_"+str(cur_epoch) + ".pt")
        # torch.save(self.global_model, global_path)
      
        client_path = os.path.join(model_path, self.model_name + self.topk_algo + str(self.k1)  + "_client_"+ str(cur_epoch) + ".pt")
        torch.save(self.clients[0].model, client_path)
        
        

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if len(self.rs_test_acc):
            algo = algo + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, "w") as hf:
                hf.create_dataset("rs_test_acc", data=self.rs_test_acc)
                hf.create_dataset("rs_test_auc", data=self.rs_test_auc)
                hf.create_dataset("rs_train_loss", data=self.rs_train_loss)

    def client_test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc 
    
    def client_evaluate(self,acc=None,loss=None):
        stats = self.client_test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
    
        return train_loss, test_acc, test_auc
  
    def global_test_metrics(self):
        testloaderfull = self.test_loader
        
        self.global_model.eval()
       
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.global_model(x)
               
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
     
        auc = metrics.roc_auc_score(y_true, y_prob, average="micro")

        return test_acc, test_num, auc

    def train_metrics(self):
        num_samples = []
        losses = []

        for c in self.clients:
            cl, ns = c.train_loss, c.train_num
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def global_evaluate(self, acc=None, loss=None):
        test_acc, test_num, test_auc = self.global_test_metrics()
        
        stats_train = self.train_metrics()
        test_acc /= test_num

        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))

        return train_loss, test_acc, test_auc

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))



