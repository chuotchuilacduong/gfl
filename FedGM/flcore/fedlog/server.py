import torch
from flcore.fedlog.utils import average_parameters, merge_graphs_ht_proto
import copy
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from flcore.base import BaseServer

class FedLogServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device, personalized=False):
        super().__init__(args, global_data, data_dir, message_pool, device, personalized)
        
        self.cache = None
        self.received_keys = {
            'client_weights', 'NeighGen_pre_weights', 'cd_graphs', 
            'cd_scaler', 'train_results', 'test_results', 'valid_results'
        }
        self.client_cd_graphs_cache = {}
        self.cd_graphs = []

    def execute(self):
        collected_messages = []
        for cid in range(self.args.num_clients):
            msg_key = f"client_{cid}"
            if msg_key in self.message_pool:
                msg = self.message_pool[msg_key]
                msg['__cid__'] = cid 
                collected_messages.append(msg)
        
        if not collected_messages:
            return

        self.update(self.device, collected_messages)

        self.send_message()
    
    def get_parameters(self):
        return 0

    def send_message(self):
        for cid in range(self.args.num_clients):
            message = {}
            if hasattr(self, 'global_weights'):
                message['server_weights'] = self.global_weights
            if hasattr(self, 'global_synthetic_data'):
                message['global_synthetic_data'] = self.global_synthetic_data
            if cid in self.client_cd_graphs_cache:
                message['my_cd_graph'] = self.client_cd_graphs_cache[cid]
            
            if hasattr(self, 'cd_graphs') and self.cd_graphs:
                message['cd_graphs'] = self.cd_graphs
            
            self.message_pool[f"server_to_client_{cid}"] = message

    def update(self, gpu, collected_messages):
        weights = [msg['client_weights'] for msg in collected_messages if 'client_weights' in msg]        
        weights = list(zip(*weights))
        if len(weights) > 0:
            global_params = average_parameters(weights)
            self.global_weights = (global_params[0], global_params[1], global_params[2], global_params[3])
        
        # ____________________________________
        # generating global synthetic features
        self.client_cd_graphs_cache = {}
        for msg in collected_messages:
            if 'cd_graphs' in msg and '__cid__' in msg:
                self.client_cd_graphs_cache[msg['__cid__']] = msg['cd_graphs']
        
        self.cd_graphs = [msg['cd_graphs'] for msg in collected_messages if 'cd_graphs' in msg]
        cd_graphs = self.cd_graphs 
        
        num_cls = self.args.num_classes        
        
        syn_xs = []
        syn_ys = []
        permuted_cd_graphs = []
        
        n_clients = len(cd_graphs)
        if n_clients > 0:
            local_cd_graphs = copy.deepcopy(cd_graphs)
            for cd_graph_cid in local_cd_graphs:
                cd_graph_x = cd_graph_cid.x_head.reshape([num_cls, self.args.num_proto, -1])
                # Permute logic
                cd_graph_cid.x_head = cd_graph_x[:, torch.randperm(cd_graph_x.size()[1]), :].reshape([num_cls * self.args.num_proto, -1])
                permuted_cd_graphs.append(cd_graph_cid)
            merged_global_graph = merge_graphs_ht_proto(permuted_cd_graphs)
            merged_global_graph = merged_global_graph.to(gpu)

            global_syn = merged_global_graph.x.view([n_clients, num_cls, self.args.num_proto, -1])
            global_cls_rate = (merged_global_graph.cls_rate.view([n_clients,-1]) / merged_global_graph.cls_rate.view([n_clients,-1]).sum(0)).view([n_clients,num_cls,1,1])
            global_cls_rate = torch.nan_to_num(global_cls_rate, nan=1/n_clients)

            weighted_global_syn = (global_syn * global_cls_rate).sum(0) 
            weighted_global_syn = weighted_global_syn.view([-1, weighted_global_syn.shape[-1]])
            weighted_global_syn_y = merged_global_graph.y[:num_cls * self.args.num_proto]

            syn_xs.append(weighted_global_syn.detach().tolist())
            syn_ys.append(weighted_global_syn_y.detach().tolist())
            self.global_synthetic_data = Data(x=syn_xs, y=syn_ys)

        gen_weights = [msg['NeighGen_pre_weights'] for msg in collected_messages if 'NeighGen_pre_weights' in msg]
        if len(gen_weights) > 0 and cd_graphs:
            global_cls_gen = []
            with torch.no_grad() :
                merged_global_graph = merge_graphs_ht_proto(cd_graphs, 'head', 'proto')
                gen_weights = list(zip(*gen_weights))

                for cls_label in range(len(merged_global_graph.y.unique())) : 

                    cls_gen_weights = copy.deepcopy(gen_weights)

                    class_indices = (merged_global_graph.y == cls_label).nonzero(as_tuple=True)[0]
                    client_cls_rate = merged_global_graph.cls_rate[class_indices]
                    client_cls_rate = client_cls_rate / (client_cls_rate).sum().cpu().tolist()
                    client_cls_rate = client_cls_rate.numpy()
                    
                    global_params = []

                    if self.cache == None :
                        sum_params = [torch.zeros_like(torch.Tensor(param)).cpu() for param in cls_gen_weights]
                        self.cache = sum_params

                    for i, params in enumerate(cls_gen_weights):
                        if len(np.array(params).shape) == 2:
                            sum_params[i] = (client_cls_rate.reshape(len(client_cls_rate),1) * np.array(params)).sum(0)
                        if len(np.array(params).shape) == 3:
                            sum_params[i] = (client_cls_rate.reshape(len(client_cls_rate),1,1) * np.array(params)).sum(0)
                    global_cls_gen.append(sum_params)
                
                sum_params = [torch.zeros_like(torch.Tensor(param)).cpu() for param in gen_weights]
                for i, params in enumerate(gen_weights):
                    sum_params[i] = np.array(params).mean(0)
                global_cls_gen.append(sum_params)
                
            self.Server_NeighGen_cls_weights = global_cls_gen
            
    def get_message_for_client(self, client_id, cur_round):
        message = {}
        if hasattr(self, 'global_weights'):
            message['server_weights'] = self.global_weights
        if hasattr(self, 'global_synthetic_data'):
            message['global_synthetic_data'] = self.global_synthetic_data
        return message