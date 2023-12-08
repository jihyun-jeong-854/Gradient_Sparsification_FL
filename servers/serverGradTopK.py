import time
from clients.clientGradTopK import clientGradTopK
from servers.server import Server

import wandb
import torch
from tqdm import tqdm

from torch import nn
class FedTopK(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        # set logger
        config = {
            "optimizer": "TOPK SGD",
            "epochs": self.global_rounds,
            "batch size": self.batch_size,
            "lr": self.learning_rate,
            "top k method": self.topk_algo,
            "k": self.topk,
            "num clients": self.num_clients,
            "model": "mobilenet_v2",
            "dataset": self.dataset,
        }
        logger = wandb.init(
            project="Fed-TopK", config=config, name="lenet_topk"
        )
       
      
        self.set_clients(clientGradTopK)
        self.init_gradients()
        print(f"total clients: {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.client_gradients = []
    
    def init_gradients(self):
        for param in self.global_model.parameters():
            param.grad = torch.zeros_like(param.data)  
          
    
    def train(self):
       
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            
            # self.send_models() # send global model to local models
            # self.updated_clients = []
            self.client_gradients = []
           
            for client in tqdm(self.clients, desc = 'training clients...'):
                client.train()
                client_gradient = client.generate_message()
                self.client_gradients.append(client_gradient)
                # self.client_gradients.append(client_gradient)
                
            self.aggregate_gradients()
            self.send_gradients()
            self.save_global_model()
            self.Budget.append(time.time() - s_t)
            
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                train_loss, test_acc, test_auc = self.client_evaluate()
               
                wandb.log(
                    {
                        "train epochs": i + 1,
                        "time cost for each epoch": self.Budget[-1],
                        "averaged local train loss": train_loss,
                        "global model test accuracy": test_acc,
                        "global model test auc": test_auc,
                    }
                )
           
            
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])


        print("\nBest accuracy.")
        self.print_(max(self.rs_test_acc), max(
            self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()     
    
    def aggregate_gradients(self):
        # self.global_model.load_state_dict(self.clients[0].model.state_dict())

        for layer,param in self.global_model.named_parameters():
            weighted_layer =  torch.zeros_like(param.data)
            for i in range(len(self.client_gradients)):
                weighted_layer += self.client_gradients[i][layer] * self.client_weights[i]
       
            param.grad.data.copy_(weighted_layer)
                
    def send_gradients(self):
        gradient = {name: params.grad.clone() for name , params in self.global_model.named_parameters()}
        for client in self.clients:
            client.set_gradient(gradient)