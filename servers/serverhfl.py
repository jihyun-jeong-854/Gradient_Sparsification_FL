import time
from clients.clientGradTopK import clientGradTopK
from servers.server import Server
from edge import Edge
import wandb
import torch
from tqdm import tqdm
from torch import nn
import numpy as np
import copy 

class HierFL(Server):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.num_edges = args.num_edges
      
        self.initialize_edges()
        self.set_clients(clientGradTopK)
        self.initialize_edges()
        print("Finished creating server and clients.")
        
        self.Budget = []
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.clock = []
        self.edges = []
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
        
    def initialize_edges(self):

        cids = np.arange(self.num_clients)
        clients_per_edge = int(self.num_clients / self.num_edges)
        for i in range(self.num_edges):
            #Randomly select clients and assign them
            selected_cids = np.random.choice(cids, clients_per_edge, replace=False)
            cids = list (set(cids) - set(selected_cids))
            self.edges.append(Edge(id = i,
                              cids = selected_cids,
                              shared_gradients = copy.deepcopy(self.clients[0].model.state_dict())))
            
            [self.edges[i].client_register(self.clients[cid]) for cid in selected_cids]
            self.edges[i].all_trainsample_num = sum(self.edges[i].sample_registration.values())
            self.edges[i].refresh_edgeserver()
            
    def train(self):
        
        for i in range(self.num_server_aggregation + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            # self.send_models() # send global model to local models
            # self.updated_clients = []
            self.client_gradients = []
            edges = []
            for num_edgeagg in range(self.num_edge_aggregtaion):
                edge_loss = [0.0]* self.num_edges
                edge_sample = [0]* self.num_edges
                correct_all = 0.0
                total_all = 0.0
                for i,edge in self.edges:
                    edge.refresh_edgeserver()
                    client_loss = 0.0
                    selected_cnum = max(int(self.clients_per_edge * args.frac),1)
                    selected_cids = np.random.choice(edge.cids,
                                                 selected_cnum,
                                                 replace = False)
                for selected_cid in selected_cids:
                    edge.client_register(self.clients[selected_cid])
                for cid in edge.cids:
                    edge.send_to_client(self.clients[cid])
                    self.clients[cid].sync_with_edgeserver()
                for client in tqdm(self.selected_cids, desc="training clients..."):
                    client.train()
                    client_gradient,topk_time_cost = client.generate_message()
                    client_loss += self.clients[selected_cid]
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
                        # "time cost for each epoch": self.Budget[-1],
                        "time cost for top k": topk_time_cost,
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
        received_gradients = [dict for dict in self.receiver_buffer]
        self.shared_gradients = {}
        for layer,param in self.global_model.named_parameters():
            weighted_layer = torch.zeros_like(param.data)
            for i in range(len(received_gradients)):
                weighted_layer = torch.add(weighted_layer,received_gradients[i][layer],alpha= self.client_weights[i])
  
            #     print(torch.count_nonzero(self.client_gradients[i][layer]))
            # print('aggregated', torch.count_nonzero(weighted_layer))
            # print("="*50)
            self.shared_gradients[layer] = weighted_layer.clone()
        return None


    def refresh_cloudserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def edge_register(self, edge):
        self.id_registration.append(edge.id)
        self.sample_registration[edge.id] = edge.all_trainsample_num
        return None

    def receive_from_edge(self, edge_id, eshared_state_dict):
        self.receiver_buffer[edge_id] = eshared_state_dict
        return None

    def aggregate(self, args):
        received_dict = [dict for dict in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        self.shared_gradients = average_weights(w=received_dict,
                                                 s_num=sample_num)
        return None

    def send_to_edge(self, edge):
        edge.receive_from_cloudserver(copy.deepcopy(self.shared_state_dict))
        return None
