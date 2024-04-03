# The structure of the edge server
# THe edge should include following funcitons
# 1. Server initialization
# 2. Server receives updates from the client
# 3. Server sends the aggregated information back to clients
# 4. Server sends the updates to the cloud server
# 5. Server receives the aggregated information from the cloud server

import copy
import torch
import numpy as np
import time
import math
from collections import defaultdict
from diffprivlib.mechanisms import Gaussian
from clients.clienthfl import reshape_gradient


class Edge:

    def __init__(self, id, cids, k1, k2, shared_gradient, dp, device):
        """
        id: edge id
        cids: ids of the clients under this edge
        receiver_buffer: buffer for the received updates from selected clients
        shared_state_dict: state dict for shared network
        id_registration: participated clients in this round of traning
        sample_registration: number of samples of the participated clients in this round of training
        all_trainsample_num: the training samples for all the clients under this edge
        shared_state_dict: the dictionary of the shared state dict
        clock: record the time after each aggregation
        :param id: Index of the edge
        :param cids: Indexes of all the clients under this edge
        :param shared_layers: Structure of the shared layers
        :return:
        """
        self.id = id
        self.cids = cids
        self.shared_gradients = shared_gradient
        self.k1 = k1
        self.k2 = k2
        self.dp = dp
        self.device = device
        self.gradient_size = len(
            torch.cat([grad.reshape(-1) for grad in self.shared_gradients.values()])
        )
        self.chunk_size = self.gradient_size // self.k1
        self.num_chunks = math.ceil(self.gradient_size / self.chunk_size)
        
        self.receiver_buffer = {}
        self.chunk_buffer = defaultdict(set)

        self.id_registration = []
        self.sample_registration = {}
        self.all_trainsample_num = 0
        self.p_clients = {}
        self.clock = []

        self.dp_gaussian = Gaussian(epsilon=1.0, delta=0.00001, sensitivity=1.0)

    def refresh_edgeserver(self):
        self.receiver_buffer.clear()
        self.chunk_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def client_register(self, client):
        self.id_registration.append(client.id)
        self.sample_registration[client.id] = client.train_data_size
        return None

    def receive_from_client(self, client_id, topk_gradient, chunk_ids):
        self.receiver_buffer[client_id] = topk_gradient
        [self.chunk_buffer[chunk_id].add(client_id) for chunk_id in chunk_ids]

        return None

    def print_chunk_buffer(self):
        sorted_dict = sorted(
            self.chunk_buffer.items(), key=lambda item: len(item[1]), reverse=True
        )
        print(sorted_dict)
        print("계산해야할 chunk 개수", len(self.chunk_buffer.keys()))

    def percentage_clients_data(self):
        for cid, sample_num in self.sample_registration.items():
            self.p_clients[cid] = sample_num / self.all_trainsample_num
        return None

    def aggregate_gradients(self):
        for layer in self.receiver_buffer[self.cids[0]].keys():
            weighted_layer = torch.zeros_like(self.receiver_buffer[self.cids[0]][layer])
            for cid in self.cids:
                weighted_layer = torch.add(
                    weighted_layer,
                    self.receiver_buffer[cid][layer],
                    alpha=self.p_clients[cid],
                )

            self.shared_gradients[layer] = weighted_layer.clone()
        return None

    def aggregate_gradients_by_chunk(self):
        zeroedout_gradient = torch.zeros(self.gradient_size, dtype=torch.float32).to(
            self.device
        )
       
        for cid in self.cids:
            # self.count_non_zero_grads(self.receiver_buffer[cid])
            self.receiver_buffer[cid] = torch.cat(
                [
                    layer.clone().reshape(-1)
                    for layer in self.receiver_buffer[cid].values()
                ]
            )


        # 전체 chunk 중에
        for chunk in range(self.num_chunks):
            start_index, end_index = self.chunkId_2_gradId(chunk)

            chunk_size = end_index - start_index
            current_chunk = torch.zeros(chunk_size, dtype=torch.float32).to(self.device)
            if len(self.chunk_buffer[chunk]) > 0:
                client_ids = self.chunk_buffer[chunk]
                
                for cid in client_ids:
               
                    current_chunk = torch.add(
                        current_chunk,
                        self.receiver_buffer[cid][start_index:end_index],
                        alpha=self.p_clients[cid],
                    )
                    
                topk = current_chunk.abs().topk(k=self.k2)
                topk_indices = start_index + topk.indices.item()
                # topk_indices = [start_index + int(id.item()) for id in topk.indices]
                zeroedout_gradient[topk_indices] = topk.values.item()
                
                # zeroedout_gradient[start_index:end_index]=current_chunk
 
        self.shared_gradients = self.reshape_gradient(zeroedout_gradient)
        self.chunk_buffer.clear()
        return None

    def add_gaussian_noise(self,shared_gradient):
        noised_params = {}
        f = lambda x: self.dp_gaussian.randomise(value=x)
        for layer, value in shared_gradient.items():
            noised_params[layer] = torch.from_numpy(np.vectorize(f)(value.cpu())).to(
                "cuda"
            )
        return noised_params
    
    def send_to_client(self, client):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        client.receive_from_edgeserver(copy.deepcopy(self.shared_gradients))

        return None

    def send_to_server(self, server):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        s_t = time.time()
        time_cost = 0
        shared_gradient = copy.deepcopy(self.shared_gradients) 
        # print('-------------------')
        # self.count_non_zero_grads(self.shared_gradients)
        if self.dp == True:
      
            shared_gradient = self.add_gaussian_noise(shared_gradient)
            time_cost = time.time() - s_t
        server.receive_from_edge(edge_id=self.id, eshared_gradients=shared_gradient)
        return time_cost

    def receive_from_server(self, shared_gradients):
        self.shared_gradients = shared_gradients
        return None

    def chunkId_2_gradId(self, current_chunk_id):
        start_index = self.chunk_size * current_chunk_id
        end_of_chunk = start_index + self.chunk_size
      
        end_index = (
            self.gradient_size
            if self.gradient_size - end_of_chunk < self.chunk_size
            else end_of_chunk
        )

        return start_index, end_index

    def reshape_gradient(self, flat_params):
        start_idx = 0
        top_k_params = {}
        for name, param in self.shared_gradients.items():
            end_idx = start_idx + param.data.numel()
        
            top_k_params[name] = flat_params[start_idx:end_idx].view(param.data.shape)
            start_idx = end_idx
       
        # self.count_non_zero_grads(top_k_params)
        return top_k_params

    def count_non_zero_grads(self,shared_gradients):
        grads = torch.cat([grad.reshape(-1) for grad in shared_gradients.values()])
        self.non_zero_grads = torch.count_nonzero(grads)
        print('non zero gradients: ', self.non_zero_grads)     
        print('top k ', torch.topk(grads,k=50).values)