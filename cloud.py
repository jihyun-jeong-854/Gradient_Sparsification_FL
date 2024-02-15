# The structure of the server
# The server should include the following functions:
# 1. Server initialization
# 2. Server reveives updates from the user
# 3. Server send the aggregated information back to clients
import numpy as np

import copy
from servers.serverGradTopK import aggregate_gradients
from clients.clientGradTopK import clientGradTopK
from edge import Edge

class Cloud(object):
    def __init__(self, args):
        self.args = args
        self.num_clients = args.num_clients
        self.num_edges = args.num_edges
        self.num_writers = args.num_writers
        self.epochs = args.epoch
        self.clients = []
        self.client_ids = []
        self.client_weights = []
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        # self.shared_state_dict = shared_layers.state_dict()
        self.clock = []
        
        self.set_clients(clientObj=clientGradTopK)
        self.assign_clients()
        print("Finished creating and assigning clients to edge.")
        
        
    def set_clients(self,clientObj):

        self.total_samples = 0
        for i in range(self.num_clients) :
            client = clientObj(
                self.args,
                id=i,
                train_samples=self.num_writers,
            )
            self.clients.append(client)
            self.client_ids.append(i)
      
    def assign_clients(self):
        self.edges = []
        cids = np.arange(self.num_clients)
        clients_per_edge = int(self.num_clients / self.num_edges)
        p_clients = [0.0] * self.num_edges
        
        for i in range(self.num_edges):
            #Randomly select clients and assign them
            selected_cids = np.random.choice(cids, clients_per_edge, replace=False)
            cids = list (set(cids) - set(selected_cids))
            self.edges.append(Edge(id = i,
                              cids = selected_cids))
            [self.edges[i].client_register(self.client_ids[cid]) for cid in selected_cids]
            self.edges[i].all_trainsample_num = sum(self.edges[i].sample_registration.values())
            p_clients[i] = [sample / float(self.edges[i].all_trainsample_num) for sample in
                    list(self.edges[i].sample_registration.values())]
            self.edges[i].refresh_edgeserver()
    
    def train(self):
        self.refresh_cloudserver()
        for num_edgeagg in range():
            edge_loss = [0.0]* self.num_edges
            edge_sample = [0]* self.num_edges
            correct_all = 0.0
            total_all = 0.0
            # no edge selection included here
            # for each edge, iterate
            for i,edge in enumerate(self.edges):
                edge.refresh_edgeserver()
                client_loss = 0.0
                selected_cnum = max(int(clients_per_edge * args.frac),1)
                selected_cids = np.random.choice(edge.cids,
                                                 selected_cnum,
                                                 replace = False,
                                                 p = p_clients[i])
                for selected_cid in selected_cids:
                    edge.client_register(clients[selected_cid])
                for selected_cid in selected_cids:
                    edge.send_to_client(clients[selected_cid])
                    clients[selected_cid].sync_with_edgeserver()
                    client_loss += clients[selected_cid].local_update(num_iter=args.num_local_update,
                                                                      device = device)
                    clients[selected_cid].send_to_edgeserver(edge)
                edge_loss[i] = client_loss
                edge_sample[i] = sum(edge.sample_registration.values())

                edge.aggregate(args)
                correct, total = all_clients_test(edge, clients, edge.cids, device)
                correct_all += correct
                total_all += total
            # end interation in edges
            all_loss = sum([e_loss * e_sample for e_loss, e_sample in zip(edge_loss, edge_sample)]) / sum(edge_sample)
            avg_acc = correct_all / total_all       
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
        self.shared_gradients = aggregate_gradients(w=received_dict,
                                                 s_num=sample_num)
        return None

    def send_to_edge(self, edge):
        edge.receive_from_cloudserver(copy.deepcopy(self.shared_state_dict))
        return None