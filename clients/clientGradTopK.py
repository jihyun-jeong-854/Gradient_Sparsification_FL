import torch
import torch.nn as nn
import numpy as np
import time
from clients.client import Client
import copy


class clientGradTopK(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)

        self.train_loss = 0
        self.train_num = 0
        self.model = copy.deepcopy(args.model)
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=0.995)
                                   
        self.accumulated_gradients = {}
        self.current_gradients = {}
        self.init_gradients()
           
    def init_gradients(self):
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)   
             
    def train(self):
        self.train_loss = 0
        self.train_num = 0
        
        self.accumulated_gradients = {
            name: torch.zeros_like(param.data) for name, param in self.model.named_parameters()
        }
        self.model.train()
        start_time = time.time()
        self.save_current_gradients()
        self.save_current_params()
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

    def accumulate_gradient(self):
        for layer_name, param in self.model.named_parameters():
            self.accumulated_gradients[layer_name] += param.grad.data.clone()
            
    def save_current_gradients(self):
        self.current_gradients = {
            name: param.grad.data.clone() for name, param in self.model.named_parameters()
        }
    def save_current_params(self):
        self.current_params = {
            name: param.data.clone() for name, param in self.model.named_parameters()
        }    
    def generate_message(self):
        # params_diff = self.subtract_params()
        # gradients = {name: params.grad.clone() for name , params in self.model.named_parameters()}
        top_k_,time_cost = self.get_top_k()  # current - initial 의 top k
        return top_k_,time_cost

    def get_top_k(self):
        # grads = {name: params.grad.clone() for name , params in self.model.named_parameters()}
     
        if self.topk_algo == "global":
            top_k,time_cost = self.global_topk(self.accumulated_gradients)
        elif self.topk_algo == "chunk":
            top_k,time_cost = self.chunk_topk(self.accumulated_gradients)
        elif self.topk_algo == "none":
            top_k = self.accumulated_gradients
            time_cost = 0
        else:
            raise NotImplementedError

        return top_k,time_cost

    def chunk_topk(self, params_diff):
        s_t = time.time()
        time_cost = 0
        all_params = torch.cat([grad.reshape(-1) for grad in params_diff.values()])
        indices = len(all_params) / self.topk
        chunks = all_params.chunk(self.topk * 2, dim=-1)
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
        return top_k_params, time_cost

    def global_topk(self, params_diff):
        s_t = time.time()
        all_grads = torch.cat([grad.reshape(-1) for grad in params_diff.values()])
        top_k = all_grads.abs().topk(self.topk)
        mask = set(range(len(all_grads))) - set(top_k.indices.tolist())

        all_grads[list(mask)] = 0
        e_t = time.time()

        start_idx = 0
        top_k_grads = {}

        for layer, param in self.model.named_parameters():
            end_idx = start_idx + param.data.numel()
            top_k_grads[layer] = all_grads[start_idx:end_idx].view(param.data.shape)
            start_idx = end_idx
    
        return top_k_grads, e_t - s_t

    def set_gradient(self, avg_gradient):

        for layer, param in self.model.named_parameters():
            param.data.copy_(self.current_params[layer])
            param.grad.data.copy_(avg_gradient[layer])
            
        self.optimizer.step()
        
        # after = torch.cat([param.data.clone().reshape(-1) for param in self.model.parameters()])      
        # print(torch.sum(before == after), len(before))
   
    # def set_gradient(optim, server_gradient):
    #     for group in optim.param_groups:
    #         for param in group['params']:
    #             if param.grad is not None:  # Ensure gradient exists
    #                 print(param.name,server_gradient)
    #                 print('='*50)
    #                 param.grad.data = server_gradient[param.name]
    
    def send_to_edgeserver(self, edgeserver):
        edgeserver.receive_from_client(client_id= self.id,
                                        cshared_state_dict = copy.deepcopy(self.model.shared_layers.state_dict())
                                        )
        return None

    def receive_from_edgeserver(self, shared_state_dict):
        self.receiver_buffer = shared_state_dict
        return None

    def sync_with_edgeserver(self):
        """
        The global has already been stored in the buffer
        :return: None
        """
        # self.model.shared_layers.load_state_dict(self.receiver_buffer)
        self.model.update_model(self.receiver_buffer)
        return None