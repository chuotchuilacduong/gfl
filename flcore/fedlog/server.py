import torch
import copy
import numpy as np
from flcore.base import BaseServer
from torch_geometric.data import Data

from .utils import average_parameters, merge_graphs_ht_proto

class FedLogServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedLogServer, self).__init__(args, global_data, data_dir, message_pool, device)
        
        self.server_weights = None
        self.global_synthetic_data = None
        self.merged_global_graph = None 
        self.cache_gen_weights = None
    def execute(self):

        weights = []
        sampled_client_indices = self.message_pool["sampled_clients"]
        current_round = self.message_pool["round"]
        for client_id in sampled_client_indices:
            client_msg = self.message_pool[f"client_{client_id}"]
            if client_msg and "weight" in client_msg:
                weights.append(client_msg["weight"])
        
        if weights:
            weights_zipped = list(zip(*weights))
            self.server_weights = average_parameters(weights_zipped)

        if current_round ==1:
            gen_weights_list = []
            cd_graphs_for_gen = []
            
            for client_id in sampled_client_indices:
                msg = self.message_pool[f"client_{client_id}"]
                if msg and "neigh_gen_weights" in msg:
                    gen_weights_list.append(msg["neigh_gen_weights"])
                if msg and "cd_graph" in msg:
                    cd_graphs_for_gen.append(msg["cd_graph"])

            if gen_weights_list and cd_graphs_for_gen:
                self.server_neigh_gen_cls_weights = []
                with torch.no_grad():
                    
                    merged_graph_gen = merge_graphs_ht_proto(cd_graphs_for_gen)
                    
                    # Transpose list of weights: [Client1_Params, Client2_Params...] -> [(Param1_C1, Param1_C2...), ...]
                    gen_weights_zipped = list(zip(*gen_weights_list))
                    
                    unique_classes = merged_graph_gen.y.unique()
                    num_unique_cls = len(unique_classes)

                    # A. Class-Specific Generators
                    for cls_label in range(num_unique_cls):
                        cls_gen_params = copy.deepcopy(gen_weights_zipped) # List of tuples of params
                        
                        num_clients_batch = len(gen_weights_list)
                        stride = len(merged_graph_gen.cls_rate) // num_clients_batch
                        
                        rate_indices = torch.arange(num_clients_batch) * stride + cls_label
                        rate_indices = rate_indices.to(merged_graph_gen.cls_rate.device)                         
                        client_cls_rate = merged_graph_gen.cls_rate[rate_indices]
                        
                        # Normalize rate
                        client_cls_rate = client_cls_rate / (client_cls_rate.sum() + 1e-10)
                        client_cls_rate = client_cls_rate.cpu().numpy()

                        aggregated_params = []
                        if self.cache_gen_weights is None:
                            self.cache_gen_weights = [torch.zeros_like(torch.tensor(p[0])).float().cpu() for p in cls_gen_params]

                        for i, params_tuple in enumerate(cls_gen_params):
                            
                            params_stack = np.array([p for p in params_tuple])                            
                            rate_reshaped = client_cls_rate.reshape((len(client_cls_rate),) + (1,) * (params_stack.ndim - 1))
                            
                            weighted_sum = (params_stack * rate_reshaped).sum(axis=0)
                            aggregated_params.append(weighted_sum)
                        
                        self.server_neigh_gen_cls_weights.append(aggregated_params)
                    avg_params = []
                    for params_tuple in gen_weights_zipped:
                         params_stack = np.array([p for p in params_tuple])
                         avg_params.append(params_stack.mean(axis=0))
                    self.server_neigh_gen_cls_weights.append(avg_params)
        
        cd_graphs = []
        for client_id in sampled_client_indices:
            client_msg = self.message_pool[f"client_{client_id}"]
            if client_msg and "cd_graph" in client_msg and client_msg["cd_graph"] is not None:
                cd_graphs.append(client_msg["cd_graph"])
        
        if cd_graphs:
            num_cls = cd_graphs[0].y.max().item() + 1
            num_clients = len(cd_graphs)

            # a. Permute & Merge Graphs
            permuted_cd_graphs = []
            for cd_graph_cid in cd_graphs:
                
                cd_graph_x = cd_graph_cid.x_head.reshape([num_cls, self.args.num_proto, -1])
                
                # Random permutation 
                # cd_graph_x[:, torch.randperm(cd_graph_x.size()[1]), :]
                perm_indices = torch.randperm(cd_graph_x.size()[1])
                shuffled_x = cd_graph_x[:, perm_indices, :].reshape([num_cls * self.args.num_proto, -1])
                
                cd_graph_copy = cd_graph_cid.clone()
                cd_graph_copy.x_head = shuffled_x
                permuted_cd_graphs.append(cd_graph_copy)


            self.merged_global_graph = merge_graphs_ht_proto(permuted_cd_graphs)
            self.merged_global_graph = self.merged_global_graph.to(self.device)

            # b. Calculate Weighted Global Synthetic Features
            global_syn = self.merged_global_graph.x.view([num_clients, num_cls, self.args.num_proto, -1])
            
            # Normalize Class Rate
            cls_rates = self.merged_global_graph.cls_rate.view([num_clients, -1])
            sum_rates = cls_rates.sum(0) 
            client_contribution = cls_rates / (cls_rates.sum(0, keepdim=True) + 1e-10) 
            
            global_cls_rate = client_contribution.view([num_clients, num_cls, 1, 1])
            global_cls_rate = torch.nan_to_num(global_cls_rate, nan=1.0/num_clients)

            weighted_global_syn = (global_syn * global_cls_rate).sum(0) 
            weighted_global_syn = weighted_global_syn.view([-1, weighted_global_syn.shape[-1]])
            
            weighted_global_syn_y = self.merged_global_graph.y[:num_cls * self.args.num_proto]

            self.global_synthetic_data = Data(x=weighted_global_syn.detach().cpu(), 
                                              y=weighted_global_syn_y.detach().cpu())
            
            self.merged_global_graph = self.merged_global_graph.cpu()

    def send_message(self):

        neigh_gen_cls_weights = getattr(self, "server_neigh_gen_cls_weights", None)
        self.message_pool["server"] = {
            "weight": self.server_weights,
            "global_synthetic_data": self.global_synthetic_data,
            "merged_global_graph": self.merged_global_graph,
            "server_neigh_gen_cls_weights": neigh_gen_cls_weights
        }