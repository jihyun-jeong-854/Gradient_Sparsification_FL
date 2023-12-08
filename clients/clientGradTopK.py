import torch
import torch.nn as nn
import numpy as np
import time
from clients.client import Client
import copy

class clientGradTopK(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.prev_model_state = None
        self.train_loss = 0
        self.train_num = 0
        self.init_gradients()
    
        
    def init_gradients(self):
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        self.accumulated_gradients = {name: params.grad.data for name, params in self.model.named_parameters()}        
    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        # self.prev_gradients = {name: params.grad.clone() for name , params in self.model.named_parameters()}
        self.train_loss = 0
        self.train_num = 0 
        self.model.train()
        self.accumulated_gradients = {}
        start_time = time.time()
        
        for i, (x, y) in enumerate(trainloader):

            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = self.loss(output, y)
            self.train_loss += loss.item() * y.shape[0]
            self.train_num += y.shape[0]
            self.optimizer.zero_grad()
            loss.backward()
            # self.optimizer.step()
            self.accumulate_gradient()    
        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def accumulate_gradient(self):
        for layer_name, param in self.model.named_parameters():
            if layer_name not in self.accumulated_gradients.keys(): 
                self.accumulated_gradients[layer_name] = param.grad.data
            else:
                self.accumulated_gradients[layer_name] += param.grad.data
            
    def generate_message(self):
        # params_diff = self.subtract_params()
        # gradients = {name: params.grad.clone() for name , params in self.model.named_parameters()}
        top_k_ = self.get_top_k()  # current - initial 의 top k
       
        return top_k_

    
    def get_top_k(self):
        # grads = {name: params.grad.clone() for name , params in self.model.named_parameters()}
        top_k = self.accumulated_gradients
        if self.topk_algo == "global":
            top_k = self.global_topk(top_k)
        elif self.topk_algo == "chunk":
            top_k = self.chunk_topk(top_k)   
        return top_k

    def chunk_topk(self,params_diff):
     
        all_params = torch.cat([grad.reshape(-1) for grad in params_diff.values()])
        chunks = all_params.chunk(self.topk, dim=-1)
        for chunk in chunks:
            local_max_index = torch.abs(chunk.data).argmax().item()
            zeroed_out = set(range(len(chunk))) - set([local_max_index])
            chunk.data[list(zeroed_out)] = 0

        top_k = torch.cat([chunk for chunk in chunks])

        start_idx = 0
        top_k_params = {}
        for name, param in self.model.named_parameters():
            end_idx = start_idx + param.data.numel()
            top_k_params[name] = top_k[start_idx:end_idx].view(param.data.shape)
            start_idx = end_idx
        return top_k_params
    
    def global_topk(self,params_diff):
       
        all_grads = torch.cat([grad.reshape(-1) for grad in params_diff.values()])
        top_k = all_grads.abs().topk(self.topk)
        mask = set(range(len(all_grads))) - set(top_k.indices.tolist())

        all_grads[list(mask)] = 0

        start_idx = 0
        top_k_grads = {}

        for layer,param in self.model.named_parameters():
            end_idx = start_idx + param.numel()
            top_k_grads[layer] = all_grads[start_idx:end_idx].view(param.data.shape)
            start_idx = end_idx
            
        return top_k_grads
   
    def set_gradient(self,avg_gradient):

        for layer,param in self.model.named_parameters():
            param.grad.data.copy_(avg_gradient[layer]) 
       
        self.optimizer.step()
        
            
        
      