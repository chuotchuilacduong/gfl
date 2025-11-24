import torch
from flcore.fedgm.server import FedGMServer
from flcore.fedgm.utils import normalize_adj_tensor
from utils.task_utils import load_node_edge_level_default_model
from model.gcn import GCN_kipf 
from flcore.fedgm.fedgm_config import config
import torch.nn.functional as F
from flcore.fedgm.utils import match_loss, regularization, tensor2onehot
import copy
class FedGMHybridServer(FedGMServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedGMHybridServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.task_model_b = load_node_edge_level_default_model(
            args, 
            input_dim=self.task.num_feats, 
            output_dim=self.task.num_global_classes
        ).to(device)
        
        self.task.load_custom_model(self.task_model_b)
        self.adj_syn_norm = None
        
    def send_message(self):
        if self.message_pool["round"] == 0:
            self.message_pool["server"] = {}
        else:
            self.message_pool["server"] = {
                "fedgm_weights": list(self.model.parameters()), 
                
                "fedavg_weights": list(self.task.model.parameters()),
                
                "syn_graph_global": {
                    "syn_x": self.syn_x,
                    "adj_syn_norm": self.adj_syn_norm, 
                    "syn_y": self.syn_y
                }
            }
            
            
    def gradient_match(self, global_class_gradient):
       
        
        syn_x, pge, syn_y = self.syn_x, self.pge, self.syn_y
        syn_class_indices = self.all_syn_class_indices

        model_parameters = list(self.model.parameters())
        self.model.train()
        adj_syn = pge(self.syn_x)
        adj_syn_norm = normalize_adj_tensor(adj_syn, sparse=False)

        loss = torch.tensor(0.0).to(self.device)
        
        for c in range(self.task.num_global_classes):
            if self.c_sum[c] == 0:
                continue
            
            # syn loss
            output_syn = self.model.forward(syn_x, adj_syn_norm)

            output_c = torch.tensor([]).to(self.device).long() 
            syn_y_c = torch.tensor([]).to(self.device).long() 
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                if c not in syn_class_indices[client_id]:
                    continue
                ind = syn_class_indices[client_id][c]
                output_c = torch.cat((output_c, output_syn[ind[0]: ind[1]]))
                syn_y_c = torch.cat((syn_y_c, syn_y[ind[0]: ind[1]]))
            
            if output_c.shape[0] == 0: 
                continue
                
            loss_syn = F.nll_loss(output_c,syn_y_c)
            gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
            
            # gradient match
            coeff = self.c_sum[c] / max(self.c_sum.values())
            global_c_loss = match_loss(gw_syn, global_class_gradient[c], config["dis_metric"], device=self.device)
            loss += coeff  * global_c_loss
        
        if config["alpha"] > 0:
            loss_reg = config["alpha"]* regularization(adj_syn, tensor2onehot(syn_y))
        else:
            loss_reg = torch.tensor(0)

        loss = loss + loss_reg

        if loss.item() < self.best_loss:
            self.best_loss = loss.item()
            self.best_syn_x = copy.deepcopy(syn_x)
            # (Bạn cũng có thể lưu best_pge_state ở đây nếu cần)
            # self.best_pge_state = copy.deepcopy(pge.state_dict()) 

        self.optimizer_feat.zero_grad()
        self.optimizer_pge.zero_grad() 
        
        loss.backward()
        
        self.optimizer_feat.step()
        self.optimizer_pge.step() 

        self.adj_syn = pge(self.syn_x).detach() 
        self.adj_syn_norm = normalize_adj_tensor(self.adj_syn, sparse=False)

        return loss
    def execute(self):
        if self.message_pool["round"] == 0:
            x_list = []
            y_list = []
            self.all_syn_class_indices = {}
            num_tot_nodes = sum([self.message_pool[f"client_{client_id}"]["num_syn_nodes"] for client_id in self.message_pool["sampled_clients"]])
            
            self.pge = self.pge.__class__(nfeat=self.task.num_feats, nnodes=num_tot_nodes, device=self.device, args=self.args).to(self.device)
            adj_syn = torch.zeros((num_tot_nodes, num_tot_nodes), dtype=torch.float32).to(self.device)

            labels = 0
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                x_list.append(self.message_pool[f"client_{client_id}"]["syn_x"])
                y_list.append(self.message_pool[f"client_{client_id}"]["syn_y"])
                local_indices = self.message_pool[f"client_{client_id}"]["local_syn_class_indices"]
                self.all_syn_class_indices[client_id] = {
                    key: (value[0] + labels, value[1] + labels)
                    for key, value in local_indices.items()
                }
                labels += self.message_pool[f"client_{client_id}"]["num_syn_nodes"]
            
            syn_x = torch.concat(x_list)
            syn_y = torch.concat(y_list)

            num_syn_all_nodes = 0
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["pge"], self.pge.parameters()):
                    global_param.data.copy_(local_param)
                
                dst_start = num_syn_all_nodes
                dst_end = num_syn_all_nodes + self.message_pool[f"client_{client_id}"]["num_syn_nodes"]
                adj_syn[dst_start:dst_end, dst_start:dst_end] = self.pge.inference(syn_x)[dst_start:dst_end, dst_start:dst_end]
                num_syn_all_nodes += self.message_pool[f"client_{client_id}"]["num_syn_nodes"]
            
            self.syn_x = syn_x.detach().requires_grad_()
            self.optimizer_feat = torch.optim.Adam([self.syn_x], lr=self.args.lr)
            self.adj_syn = adj_syn.detach()
            self.syn_y = syn_y.detach()
            self.optimizer_feat = torch.optim.Adam([self.syn_x], lr=config["lr_feat"])
            self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=config["lr_adj"])
            self.pge.train()

            self.adj_syn_norm = normalize_adj_tensor(self.adj_syn, sparse=False)
            self.model.initialize()
            self.model.fit_with_val(self.syn_x, self.adj_syn, self.syn_y, self.task.splitted_data, train_iters=600, normalize=True, verbose=False)
            self.adj_syn_norm = normalize_adj_tensor(self.adj_syn, sparse=False)
        else:
            # tong hop fedavg
            with torch.no_grad():
                num_tot_samples = sum([self.message_pool[f"client_{cid}"]["num_samples"] for cid in self.message_pool["sampled_clients"]])
                
                for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                    weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                    client_weights_b = self.message_pool[f"client_{client_id}"]["fedavg_weights"]
                    
                    for (local_param, global_param) in zip(client_weights_b, self.task.model.parameters()):
                        if it == 0:
                            global_param.data.copy_(weight * local_param)
                        else:
                            global_param.data += weight * local_param
            
            all_local_num_class_dict = {}
            all_local_class_gradient = {}
            for client_id in self.message_pool["sampled_clients"]:
                tmp = self.message_pool[f"client_{client_id}"]
                all_local_class_gradient[client_id] = tmp["fedgm_gradient"]
                all_local_num_class_dict[client_id] = tmp["local_num_class_dict"]
            
            self.global_class_gradient = {}
            c_sum = {}
            for c in range(self.task.num_global_classes):
                c_sum[c] = 0
                for client_id in self.message_pool["sampled_clients"]:
                    try:
                        c_sum[c] += all_local_num_class_dict[client_id][c]
                    except KeyError:
                        all_local_num_class_dict[client_id][c] = 0
                        c_sum[c] += all_local_num_class_dict[client_id][c]
            self.c_sum = c_sum

            for c in range(self.task.num_global_classes):
                flag = 0
                for client_id in self.message_pool["sampled_clients"]:
                    if c not in all_local_class_gradient[client_id] or c_sum[c] == 0:
                        continue
                    if flag == 0:
                        self.global_class_gradient[c] = copy.deepcopy(all_local_class_gradient[client_id][c])
                        for ig in range(len(self.global_class_gradient[c])):
                            self.global_class_gradient[c][ig] = self.global_class_gradient[c][ig] * (all_local_num_class_dict[client_id][c] / c_sum[c])
                        flag = 1
                    else:
                        for ig in range(len(self.global_class_gradient[c])):
                            self.global_class_gradient[c][ig] += (all_local_num_class_dict[client_id][c] / c_sum[c]) * all_local_class_gradient[client_id][c][ig]
            # grad matching
            gm_loss = self.gradient_match(self.global_class_gradient)
            self.last_gradient_match_loss = gm_loss.item() # Lưu loss
            
            # optim model A
            self.model.initialize()
            self.model.fit_with_val(
                self.syn_x, self.adj_syn, self.syn_y, 
                self.task.splitted_data, train_iters=600, normalize=True, verbose=False
            )
            
            self.adj_syn_norm = normalize_adj_tensor(self.adj_syn, sparse=False)
            
            
    