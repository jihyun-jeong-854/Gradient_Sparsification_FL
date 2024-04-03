import time
from clients.clienthfl import clienthfl
from servers.server import Server
from edge import Edge
import wandb
import torch
from tqdm import tqdm
from torch import nn
import numpy as np
import copy 

class HierFL(Server):
    def __init__(self, args,times):
        super().__init__(args, times)
        self.args = args
        self.num_edges = args.num_edges
        self.clients_per_edge = int(self.num_clients / self.num_edges)
        self.num_server_aggregation = args.num_server_aggregation
        self.num_edge_aggregation = args.num_edge_aggregation
        self.local_epoch = args.local_epoch
        self.dp = args.dp
        self.p_edges = []
        self.all_train_samples = 0
        self.Budget = []
        self.receiver_buffer = {}
        
        self.shared_gradients = {}
        self.id_registration = []
        self.sample_registration = {}
        self.clock = []
        self.edges = []
    
        self.topk_in_chunk = True
        self.k1 = args.k1
        self.k2 = args.k2
        
        self.set_clients(clienthfl)
        self.initialize_edges()
        print("Finished creating server and clients.")
        
        # set logger
        config = {
            "optimizer": "SGD",
            "server epochs": self.num_server_aggregation,
            "edge epochs":self.num_edge_aggregation,
            "local epochs":self.local_epoch,
            "batch size": self.batch_size,
            "lr": self.learning_rate,
            "top k method": self.topk_algo,
            "top k on client": self.k1,
            "top k on chunk": self.topk_in_chunk,
            "top k on chunk in edge": self.k2,
            "num clients": self.num_clients,
            "num edges": self.num_edges,
            "model": self.model_name,
            "dataset": self.dataset,
            "dp": self.dp
        }
        logger = wandb.init(project="HierFL_DP", config=config, name=f"{self.model_name}_{self.topk_algo}_K_{self.k1}")
        print(f"total clients: {self.num_clients}")
        
    def initialize_edges(self): 
        # p_clients = [0.0] * self.num_edges
        cids = np.arange(self.num_clients)
        self.clients_per_edge = int(self.num_clients / self.num_edges)
        for eid in range(self.num_edges):
            #Randomly select clients and assign them
            selected_cids = np.random.choice(cids, self.clients_per_edge, replace=False)
            cids = list (set(cids) - set(selected_cids))
            self.edges.append(Edge(id = eid,
                              cids = selected_cids,
                              k1 = self.k1,
                              k2 = self.k2,
                              shared_gradient = copy.deepcopy(self.clients[0].current_gradients),
                              dp = self.dp,
                              device = self.device))
            
            [self.edges[eid].client_register(self.clients[cid]) for cid in selected_cids]
            edge_train_samples = sum(self.edges[eid].sample_registration.values())
            self.edges[eid].all_trainsample_num = edge_train_samples
            self.edges[eid].percentage_clients_data()
            self.edges[eid].refresh_edgeserver()
    
            self.all_train_samples += edge_train_samples
       
        for eid in range(self.num_edges):
            self.p_edges.append(self.edges[eid].all_trainsample_num / self.all_train_samples)
            
        print('-------------Edge Info-------------')
        print('chunk size: ', self.edges[0].chunk_size)
        print('num of chunks: ', self.edges[0].num_chunks)  
          
    def train(self):
        p_clients = [0.0] * self.num_edges

        for num_comm in range(self.num_server_aggregation + 1):
            s_t = time.time()
            self.refresh_cloudserver()
            agg_time = 0
            correct_all = 0.0
            total_all = 0.0
            for num_edgeagg in range(self.num_edge_aggregation):
                edge_loss = [0.0]* self.num_edges
                edge_sample = [0]* self.num_edges
                
                for eid,edge in enumerate(self.edges):
                    edge.refresh_edgeserver()
                    client_loss = 0.0
                  
                    selected_cnum = max(int(self.clients_per_edge * self.join_ratio),1)
                    selected_cids = np.random.choice(edge.cids,
                                                 selected_cnum,
                                                 replace = False)

                    # 이번 epoch 에서 학습에 참여할 client 고른다.
                    for selected_cid in selected_cids:
                        edge.client_register(self.clients[selected_cid])
                        
                    # 현재 model state dict 보내줌 , client.receive_from _edgeserver()
                    for selected_cid in selected_cids:
                        self.clients[selected_cid].train()
                        client_loss += self.clients[selected_cid].train_loss / self.clients[selected_cid].train_num
                        # client_gradient,topk_time_cost = client.generate_message()
                        self.clients[selected_cid].send_to_edgeserver(edge)
                    
                    edge_loss[eid] = client_loss
                    edge_sample[eid] = sum(edge.sample_registration.values())
                    agg_start = time.time()
                    
                    if self.topk_in_chunk == True:
                        edge.aggregate_gradients_by_chunk()
                    else:
                        edge.aggregate_gradients()
                    # 
                    agg_time += time.time() - agg_start
                    # edge.print_chunk_buffer()
                    [edge.send_to_client(self.clients[selected_cid]) for selected_cid in selected_cids]
                    
                    # server aggregation안하는 경우만
                    if num_edgeagg < self.num_edge_aggregation - 1:
                        [self.clients[selected_cid].sync_with_edgeserver() for selected_cid in selected_cids]
                        
            noise_time = 0         
            for edge in self.edges:
                noise_time += edge.send_to_server(self)
            noise_time /= self.num_edges
            
            self.aggregate_gradients()
            self.count_non_zero_grads()
            for edge in self.edges:
                self.send_to_edge(edge)
                client_ids = edge.cids
                [edge.send_to_client(self.clients[cid]) for cid in client_ids]
                [self.clients[cid].sync_with_edgeserver() for cid in client_ids]
                
            correct, total = self.all_clients_test(edge, self.clients, edge.cids)
            correct_all += correct
            total_all += total    
            # self.save_global_model(num_comm)

            self.Budget.append(time.time() - s_t)
            if num_comm % self.eval_gap == 0:
                print(f"\n-------------Round number: {num_comm}-------------")
                print("\nEvaluate global model")
                all_loss = sum([e_loss * e_sample for e_loss, e_sample in zip(edge_loss, edge_sample)]) / sum(edge_sample)
                test_acc = correct_all / total_all
                print("train loss: ", all_loss)
                print("test_acc: ", test_acc)
                print("total time cost for agg gradient", agg_time)
                print("time cost for chunk", agg_time/(self.num_edges * self.num_edge_aggregation))
                print("time cost for add noise", noise_time)
                wandb.log(
                    {
                        "train epochs": num_comm + 1,
                        "time cost for each epoch": self.Budget[-1],
                        # "time cost for top k": topk_time_cost,
                        "averaged train loss": all_loss,
                        "averaged test accuracy": test_acc,
                        # "averaged test auc": test_auc,
                        # "number of non zero grads": self.non_zero_grads
                    }
                )

            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])

        # print("\nBest accuracy : ", max(self.rs_test_acc))
        # self.print_(
        #     sum(self.rs_test_acc)/len(self.rs_test_acc), sum(self.rs_test_auc)/len(self.rs_test_auc),sum(self.rs_train_loss)/len(self.rs_train_loss)
        # )
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()

    def refresh_cloudserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def edge_register(self, edge):
        self.id_registration.append(edge.id)
        self.sample_registration[edge.id] = edge.all_trainsample_num
        return None

    def receive_from_edge(self, edge_id, eshared_gradients):
        self.receiver_buffer[edge_id] = eshared_gradients
       
        return None

    def send_to_edge(self, edge):
        edge.receive_from_server(copy.deepcopy(self.shared_gradients))
        return None
    
    def aggregate_gradients(self):
        for layer in self.receiver_buffer[0].keys():
            weighted_layer = torch.zeros_like(self.receiver_buffer[0][layer])    
            for eid, agg_grad in self.receiver_buffer.items():
                weighted_layer = torch.add(weighted_layer,agg_grad[layer],alpha=self.p_edges[eid])
        
            self.shared_gradients[layer] = weighted_layer.clone()
          
    def count_non_zero_grads(self):
        
        grads = torch.cat([grad.reshape(-1) for grad in self.shared_gradients.values()])
        # print(torch.topk(grads,k=100))
        print(grads)
        self.non_zero_grads = torch.count_nonzero(grads)
        print('received non zero gradients: ', self.non_zero_grads)     
        
    def train_metrics(self,cids):
        num_samples = []
        losses = []

        for cid in cids:
            cl, ns = self.client[cid].train_loss, self.client[cid].train_num
            num_samples.append(ns)
            losses.append(cl * 1.0)
        return  losses
    
    def all_clients_test(self, edge, clients, cids):
        # [edge.send_to_client(clients[cid]) for cid in cids]
        # for cid in cids:
        #     edge.send_to_client(clients[cid])
        #     # The following sentence!
        #     clients[cid].sync_with_edgeserver()
        correct_edge = 0.0
        total_edge = 0.0
        for cid in cids:
            correct, total = clients[cid].test_model()
            correct_edge += correct
            total_edge += total
        return correct_edge, total_edge