from flcore.base import BaseServer
import torch
import torch.nn as nn
from flcore.base import BaseClient
from flcore.fedgm.fedgm_config import config
from flcore.fedgm.pge import PGE
from torch_geometric.utils.sparse import to_torch_sparse_tensor
import random
import numpy as np
from torch_sparse import SparseTensor
from flcore.fedgm.utils import match_loss, regularization, tensor2onehot
import torch.nn.functional as F
from torch_geometric.loader import NeighborSampler
import numpy as np
from flcore.fedgm.utils import  normalize_adj_tensor
from model.gcn import GCN_kipf
from utils.metrics import compute_supervised_metrics
import networkx as nx
import matplotlib.pyplot as plt
import copy


class FedGMServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedGMServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.model = GCN_kipf(nfeat=self.task.num_feats, 
                                             nhid=args.hid_dim, 
                                             nclass=self.task.num_global_classes, 
                                             nlayers=args.num_layers, 
                                             dropout=args.dropout, 
                                             lr=args.lr, 
                                             weight_decay=args.weight_decay, 
                                             device=self.device).to(self.device)
        self.model.initialize()
        self.pge = PGE(nfeat=self.task.num_feats, nnodes = 1, device=self.device, args=self.args).to(self.device)
       # self.task.override_evaluate = self.get_override_evaluate()

        self.best_loss = float('inf')
        self.best_syn_x = None
   
    def execute(self):
        if self.message_pool["round"] == 0:
            x_list = []
            y_list = []
            self.all_syn_class_indices = {}

            num_tot_nodes = sum([self.message_pool[f"client_{client_id}"]["num_syn_nodes"] for client_id in self.message_pool[f"sampled_clients"]])
            
            self.pge = PGE(nfeat=self.task.num_feats, nnodes = num_tot_nodes, device=self.device, args=self.args).to(self.device)
            
            adj_syn = torch.zeros((num_tot_nodes, num_tot_nodes), dtype=torch.float32).to(self.device)

            labels = 0
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                x_list.append(self.message_pool[f"client_{client_id}"]["syn_x"])
                y_list.append(self.message_pool[f"client_{client_id}"]["syn_y"])
                self.all_syn_class_indices[client_id] = self.message_pool[f"client_{client_id}"]["local_syn_class_indices"]
                self.all_syn_class_indices[client_id] = {
                    key: (value[0] + labels, value[1] + labels)
                    for key, value in self.all_syn_class_indices[client_id].items()
                }
                labels += self.message_pool[f"client_{client_id}"]["num_syn_nodes"]
            
            syn_x = torch.concat(x_list)
            syn_y = torch.concat(y_list)

            adj_link_pre = self.pge.inference(syn_x)

            num_syn_all_nodes = 0
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["pge"], self.pge.parameters()):
                    global_param.data.copy_(local_param)
                if it ==0:
                    adj_link_pre = self.pge.inference(syn_x)
                else: 
                    adj_link_pre += self.pge.inference(syn_x)
                dst_start = num_syn_all_nodes
                dst_end = num_syn_all_nodes + self.message_pool[f"client_{client_id}"]["num_syn_nodes"]
                adj_syn[dst_start:dst_end, dst_start:dst_end] = self.pge.inference(syn_x)[dst_start:dst_end, dst_start:dst_end]
                num_syn_all_nodes += self.message_pool[f"client_{client_id}"]["num_syn_nodes"]
                
            self.syn_x = syn_x.detach().requires_grad_()
            self.optimizer_feat = torch.optim.Adam([self.syn_x], lr=config["lr_feat"]) # parameterized syn_x

            self.adj_syn = adj_syn.detach()
            
            self.syn_y = syn_y.detach()
            
        else:
             
            all_local_num_class_dict = {}
            all_local_class_gradient = {}

            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                tmp = self.message_pool[f"client_{client_id}"]
                all_local_class_gradient[client_id] = tmp["local_class_gradient"]
                all_local_num_class_dict[client_id] = tmp["local_num_class_dict"]
            
            self.global_class_gradient = {}
            c_sum = {}
            for c in range(self.task.num_global_classes):
                c_sum[c] = 0
                for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                    try:
                        c_sum[c] += all_local_num_class_dict[client_id][c]
                    except KeyError:
                        # Key not found
                        all_local_num_class_dict[client_id][c] = 0
                        c_sum[c] += all_local_num_class_dict[client_id][c]

            self.c_sum = c_sum

            for c in range(self.task.num_global_classes):
                flag = 0
                for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                    if c not in all_local_class_gradient[client_id] or c_sum[c]==0:
                        continue
                    if flag == 0:
                        self.global_class_gradient[c] =  all_local_class_gradient[client_id][c]
                        for ig in range(len(self.global_class_gradient[c])):
                            self.global_class_gradient[c][ig] = self.global_class_gradient[c][ig] * (all_local_num_class_dict[client_id][c]/c_sum[c])
                        flag = 1
                    else:
                        for ig in range(len(self.global_class_gradient[c])):
                            self.global_class_gradient[c][ig] += (all_local_num_class_dict[client_id][c]/c_sum[c]) * all_local_class_gradient[client_id][c][ig]

            self.gradient_match(self.global_class_gradient)
        
        self.model.initialize()
        
        
        self.task.load_custom_model(GCN_kipf(nfeat=self.task.num_feats, 
                                            nhid=self.args.hid_dim, 
                                            nclass=self.task.num_global_classes, 
                                            nlayers=self.args.num_layers, 
                                            dropout=self.args.dropout, 
                                            lr =self.args.lr,
                                            weight_decay=self.args.weight_decay, 
                                            device=self.device))
        
        self.task.model.initialize()
        self.task.model.fit_with_val(self.syn_x, self.adj_syn, self.syn_y, self.task.splitted_data, train_iters=600, normalize=True, verbose=False)
        

    def send_message(self):
        if self.message_pool["round"] != 0:
            self.message_pool["server"] = {
                "weight": list(self.model.parameters()),
            }


    
    def gradient_match(self, global_class_gradient):
        syn_x, adj_syn, syn_y = self.syn_x, self.adj_syn, self.syn_y
        syn_class_indices = self.all_syn_class_indices

        model_parameters = list(self.model.parameters())
        self.model.train()

        adj_syn_norm = normalize_adj_tensor(adj_syn, sparse=False)

        loss = torch.tensor(0.0).to(self.device)
        
        # class-wise
        for c in range(self.task.num_global_classes):
            if self.c_sum[c] ==0:
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
            loss_syn = F.nll_loss(output_c,syn_y_c)
            gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
            
            # gradient match
            coeff = self.c_sum[c] / max(self.c_sum.values())
            global_c_loss = match_loss(gw_syn, global_class_gradient[c], config["dis_metric"], device=self.device)
            loss += coeff  * global_c_loss
        
        # TODO: regularize
        if config["alpha"] > 0:
            loss_reg = config["alpha"]* regularization(adj_syn, tensor2onehot(syn_y))
        else:
            loss_reg = torch.tensor(0)

        loss = loss + loss_reg

        if loss.item() < self.best_loss:
            self.best_loss = loss.item()
            self.best_syn_x = copy.deepcopy(syn_x)

        self.optimizer_feat.zero_grad()
        
        loss.backward()
        self.optimizer_feat.step()
        return loss

    def draw_graph(self,edge_index,name):
        g = nx.Graph()
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        edgelist = zip(src, dst)
        for i, j in edgelist:
            g.add_edge(i, j)        
        nx.draw(g)
        plt.savefig(name+'.png')
        plt.clf()
        g.clear()

    def get_override_evaluate(self):
        def override_evaluate(splitted_data=None, mute=False):
            if splitted_data is None:
                splitted_data = self.task.splitted_data
            else:
                names = ["data", "train_mask", "val_mask", "test_mask"]
                for name in names:
                    assert name in splitted_data
        
            
            eval_output = {}

            self.task.model.eval()
            num_nodes = splitted_data["data"].x.shape[0]
            real_adj = to_torch_sparse_tensor(splitted_data["data"].edge_index,size=(num_nodes, num_nodes))           

            with torch.no_grad():
                embedding = None
                logits = self.task.model.predict(splitted_data["data"].x, real_adj)
                loss_train = F.nll_loss(logits[splitted_data["train_mask"]],splitted_data["data"].y[splitted_data["train_mask"]])
                loss_val = F.nll_loss(logits[splitted_data["val_mask"]], splitted_data["data"].y[splitted_data["val_mask"]])
                loss_test = F.nll_loss(logits[splitted_data["test_mask"]], splitted_data["data"].y[splitted_data["test_mask"]])

            
            eval_output["embedding"] = embedding
            eval_output["logits"] = logits
            eval_output["loss_train"] = loss_train
            eval_output["loss_val"]   = loss_val
            eval_output["loss_test"]  = loss_test
            
            
            metric_train = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["train_mask"]], labels=splitted_data["data"].y[splitted_data["train_mask"]], suffix="train")
            metric_val = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["val_mask"]], labels=splitted_data["data"].y[splitted_data["val_mask"]], suffix="val")
            metric_test = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["test_mask"]], labels=splitted_data["data"].y[splitted_data["test_mask"]], suffix="test")
            eval_output = {**eval_output, **metric_train, **metric_val, **metric_test}
            
            info = ""
            for key, val in eval_output.items():
                try:
                    info += f"\t{key}: {val:.4f}"
                except:
                    continue

            prefix = "[server]"
            if not mute:
                print(prefix+info)
            return eval_output
            
        return override_evaluate
