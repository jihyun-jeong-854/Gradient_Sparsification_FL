import torch
import torch.nn as nn
import numpy as np
import time
from clients.client import Client
import copy
from utils.privacy import *


def reshape_gradient(origin_params, flat_params):
    start_idx = 0
    top_k_params = {}
    for name, param in origin_params.items():
        end_idx = start_idx + param.data.numel()
        top_k_params[name] = flat_params[start_idx:end_idx].view(param.data.shape)
        start_idx = end_idx
    return top_k_params

class clienthfl(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)

        self.train_loss = 0
        self.train_num = 0
        self.model = copy.deepcopy(args.model)
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=0.995)
        self.local_epoch = args.local_epoch
                                   
        self.accumulated_gradients = {}
        self.current_gradients = {}
        self.current_params = {}
        self.receiver_buffer = {}
        
        self.init_gradients()
        self.save_current_gradients()   
        
    def init_gradients(self):
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)   
             
    def train(self):
        self.train_loss = 0
        self.train_num = 0
        
        
        self.model.train()
        
        # differential privacy
        # model_origin = copy.deepcopy(self.model)
        # self.model, self.optimizer, self.trainloader, privacy_engine = \
        #     initialize_dp(self.model, self.optimizer, self.trainloader, self.dp_sigma)
            
        start_time = time.time()
        self.save_current_gradients()
        self.save_current_params()
        self.initilize_acc_gradient()
        for epoch in range(self.local_epoch):
            for i, (x, y) in enumerate(self.trainloader):
            # x,y = self.get_next_train_batch(self.local_epochs)
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                self.optimizer.zero_grad()
                
                loss = self.loss(output, y)
                self.train_loss += loss.item() * y.shape[0]
                self.train_num += y.shape[0]
                
                loss.backward()
                self.optimizer.step()
                self.accumulate_gradient()
                
            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()
  
        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time
        
        # eps,DELTA = get_dp_params(privacy_engine)
        # print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")
        
        # for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
        #     param.data = param_dp.data.clone()
            
        # self.model = model_origin
        # self.optimizer = torch.optim.SGD(self.model.parameters(),lr = self.learning_rate)
        
    def accumulate_gradient(self):
        for layer_name, param in self.model.named_parameters():
            # layer_name = layer_name.split('.')[1] +"."+ layer_name.split('.')[2]
            self.accumulated_gradients[layer_name] += param.grad.data.clone()
    
    def initilize_acc_gradient(self):
        for layer_name, param in self.model.named_parameters():
            self.accumulated_gradients[layer_name] = torch.zeros_like(param.data)  
            
    def save_current_gradients(self):
        for layer, param in self.model.named_parameters():
            self.current_gradients[layer] = param.grad.data.clone()
            
    def save_current_params(self):
        for layer, param in self.model.named_parameters():
            self.current_params[layer] = param.data.clone()
            
    def generate_message(self):
        # params_diff = self.subtract_params()
        # gradients = {name: params.grad.clone() for name , params in self.model.named_parameters()}
        top_k_flat,time_cost = self.get_top_k()  # current - initial 의 top k
        gradient = reshape_gradient(self.current_gradients,top_k_flat)
        return gradient,time_cost

    def get_top_k(self):
        # grads = {name: params.grad.clone() for name , params in self.model.named_parameters()}
     
        if self.topk_algo == "global":
           top_k,time_cost = self.global_topk_(self.accumulated_gradients)            
        elif self.topk_algo == "chunk":
            top_k,time_cost = self.chunk_topk(self.accumulated_gradients)
        elif self.topk_algo == "none":
            top_k = self.accumulated_gradients
            time_cost = 0
        else:
            raise NotImplementedError

        return top_k,time_cost
    

    
    def chunk_topk(self, params_diff):
        all_params = torch.cat([grad.clone().reshape(-1) for grad in params_diff.values()])
        chunk_len =int(len(all_params) / (self.topk * 2))
        mask = []
        
        s_t = time.time()
        for chunk_id in range(self.topk * 2):
            local_max_index = torch.abs(all_params[chunk_id*chunk_len:(chunk_id+1)*chunk_len]).argmax().item()
            mask.extend(set(range(chunk_id*chunk_len,(chunk_id+1)*chunk_len)) - set([chunk_id*chunk_len + local_max_index]))
            time_cost = time.time() - s_t
        all_params[list(mask)] = 0        
        return all_params, time_cost
    
    def global_topk(self, params_diff):
        s_t = time.time()
        all_grads = torch.cat([grad.clone().reshape(-1) for grad in params_diff.values()])
        top_k = all_grads.abs().topk(self.topk)
        mask = set(range(len(all_grads))) - set(top_k.indices.tolist())

        all_grads[list(mask)] = 0
        e_t = time.time()
        
        return all_grads, e_t - s_t

    def global_topk_(self, params_diff):
       
        all_grads = torch.cat([grad.clone().reshape(-1) for grad in params_diff.values()])
        chunk_len =len(all_grads) // self.topk
        top_k = all_grads.abs().topk(self.topk)
        mask = set(range(len(all_grads))) - set(top_k.indices.tolist())
        chunk_ids = [index//chunk_len for index in top_k.indices.tolist()]
        all_grads[list(mask)] = 0
       
        # print(self.id,
        #       len(set(chunk_ids)))
        # print('='*50)
        return all_grads, chunk_ids
    
    def receive_from_edgeserver(self,agg_gradient):
        for layer,param in self.model.named_parameters():
            self.receiver_buffer[layer] = agg_gradient[layer]
              
    
    def send_to_edgeserver(self, edgeserver):
        top_k , time_cost = self.generate_message()
        edgeserver.receive_from_client(self.id,top_k,time_cost)                          
        return None

    def sync_with_edgeserver(self):
        
        """
        The global has already been stored in the buffer
        :return: None
        """

        for layer, param in self.model.named_parameters():
            param.data.copy_(self.current_params[layer])
            param.grad.data.copy_(self.receiver_buffer[layer])
            
        self.optimizer.step()
       
        return None
