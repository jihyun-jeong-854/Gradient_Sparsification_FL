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
            "model": self.model_name,
            "dataset": self.dataset,
        }
        logger = wandb.init(project="Fed_TopK_New", config=config, name=f"{self.model_name}_{self.topk_algo}_K_{self.topk}")
        print(f"total clients: {self.num_clients}")
        self.set_clients(clientGradTopK)
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.client_gradients = []
        self.non_zero_grads = 0
   
    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            # self.send_models() # send global model to local models
            # self.updated_clients = []
            self.client_gradients = []

            for client in tqdm(self.selected_clients, desc="training clients..."):
                client.train()
                client_gradient = client.generate_message()
                self.client_gradients.append(client_gradient)
            
            self.aggregate_gradients()
            self.send_gradients()
            self.save_global_model(i)

            self.Budget.append(time.time() - s_t)
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                train_loss, test_acc, test_auc = self.client_evaluate()

                wandb.log(
                    {
                        "train epochs": i + 1,
                        "time cost for each epoch": self.Budget[-1],
                        "averaged train loss": train_loss,
                        "averaged test accuracy": test_acc,
                        "averaged test auc": test_auc,
                        "number of non zero grads": self.non_zero_grads
                    }
                )

            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])

        print("\nBest accuracy : ", max(self.rs_test_acc))
        self.print_(
            sum(self.rs_test_acc)/len(self.rs_test_acc), sum(self.rs_test_auc)/len(self.rs_test_auc),sum(self.rs_train_loss)/len(self.rs_train_loss)
        )
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()

    def aggregate_gradients(self):
        # self.global_model.load_state_dict(self.clients[0].model.state_dict())
        self.grads = {}
        for layer,param in self.global_model.named_parameters():
            weighted_layer = torch.zeros_like(param.data)
            for i in range(len(self.client_gradients)):
                weighted_layer = torch.add(weighted_layer,self.client_gradients[i][layer],alpha= self.client_weights[i])
  
            #     print(torch.count_nonzero(self.client_gradients[i][layer]))
            # print('aggregated', torch.count_nonzero(weighted_layer))
            # print("="*50)
            self.grads[layer] = weighted_layer.clone()

    def send_gradients(self):
        # self.count_non_zero_grads()
        for client in self.clients:
            client.set_gradient(self.grads)

    def count_non_zero_grads(self):
        grads = torch.cat([grad.reshape(-1) for grad in self.grads.values()])
        self.non_zero_grads = torch.count_nonzero(grads)
     