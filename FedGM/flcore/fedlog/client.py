import torch
import torch.nn.functional as F
import numpy as np
import copy
from flcore.base import BaseClient
from torch_geometric.data import Data, Batch
from torch_geometric.nn.models import GraphSAGE
from torch_geometric.utils import to_edge_index, remove_self_loops, degree, k_hop_subgraph
from torch_sparse import SparseTensor

from .embedder.Linear import Linear
from .embedder.classifier import clsf
from .embedder.Neighgen import neighGen
from .utils import get_parameters, set_parameters, euclidean_dist, accuracy
#TODO: the model dont update per round, minor bug with env
class FedLogClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedLogClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        
        self.local_graph = self.task.data.to(self.device)
        self.num_features = self.local_graph.x.shape[1]
        if hasattr(self.task, 'train_mask'):
            self.local_graph.train_mask = self.task.train_mask
        if hasattr(self.task, 'val_mask'):
            self.local_graph.val_mask = self.task.val_mask
        if hasattr(self.task, 'test_mask'):
            self.local_graph.test_mask = self.task.test_mask
        if hasattr(self.args, 'num_classes'):
            self.num_classes = self.args.num_classes
        else:
            self.num_classes = self.local_graph.y.max().item() + 1
            
        self.embedder = GraphSAGE(
            in_channels=self.num_features, 
            hidden_channels=args.hid_dim*2, 
            out_channels=args.hid_dim, 
            dropout=0.5, 
            num_layers=2
        ).to(self.device)
        
        self.clsf_head = clsf(args.hid_dim, args.hid_dim, n_layers=2).to(self.device)
        self.clsf_tail = clsf(args.hid_dim, args.hid_dim, n_layers=2).to(self.device)
        self.classifier = Linear(args.hid_dim, self.num_classes).to(self.device)
        
        #  Generator & Synthetic Data buffers ---
        self.global_num_classes = self.num_classes
        self.neigh_gen = neighGen(self.num_features, args).to(self.device)
        self.neigh_cls_gen = [neighGen(self.num_features, args).to(self.device) for _ in range(self.global_num_classes)]
        self.cls_cd_scaler = torch.tensor([0.0] * self.num_classes).to(self.device)

        self.cond_num_nodes_per_class_per_client = self.args.num_proto
        self.cond_num_nodes_per_client = self.cond_num_nodes_per_class_per_client * self.global_num_classes
        
        syn_y = []
        for cls in range(self.num_classes): 
            syn_y += [cls] * int(self.cond_num_nodes_per_class_per_client)
        self.syn_y = torch.LongTensor(syn_y).to(self.device)
        

        self.syn_adj = SparseTensor.eye(int(self.cond_num_nodes_per_client), int(self.cond_num_nodes_per_client)).t().to(self.device)

        self.syn_feat_head = torch.FloatTensor(int(self.cond_num_nodes_per_client), self.num_features)
        torch.nn.init.xavier_uniform_(self.syn_feat_head)
        self.syn_feat_head = torch.nn.Parameter(
            torch.FloatTensor(int(self.cond_num_nodes_per_client), self.num_features).to(self.device)
        )
        torch.nn.init.xavier_uniform_(self.syn_feat_head)

        self.syn_feat_tail = torch.nn.Parameter(
            torch.FloatTensor(int(self.cond_num_nodes_per_client), self.num_features).to(self.device)
        )
        torch.nn.init.xavier_uniform_(self.syn_feat_tail)
        self.syn_feat_tail = self.syn_feat_tail.to(self.device) 
        self.n_way = self.global_num_classes
        self.k_shot = self.cond_num_nodes_per_class_per_client

        self.global_synthetic_data = None
        self.merged_global_graph = None 
        self.optimizer = torch.optim.Adam([
            {'params': self.embedder.parameters()},
            {'params': self.classifier.parameters()},
            {'params': self.clsf_head.parameters()},
            {'params': self.clsf_tail.parameters()}],
            lr=self.args.lr, weight_decay=5e-4
        )
        self.optim_syn_head = torch.optim.Adam([self.syn_feat_head], lr=self.args.lr)
        self.optim_syn_tail = torch.optim.Adam([self.syn_feat_tail], lr=self.args.lr)
        
        self.upload_weights = None
        self.upload_cd_graph = None
        self.train_loss = 0.0
        self.train_acc = 0.0

    def execute(self):

        current_round = self.message_pool["round"]

        if current_round > 0 and self.message_pool["server"] is not None:
            server_msg = self.message_pool["server"]
            
            if "weight" in server_msg:
                weights = server_msg["weight"]
                set_parameters(self.embedder, weights[0])
                set_parameters(self.classifier, weights[1])
                set_parameters(self.clsf_head, weights[2])
                set_parameters(self.clsf_tail, weights[3])
            
            if "global_synthetic_data" in server_msg:
                self.global_synthetic_data = server_msg["global_synthetic_data"]
                if self.global_synthetic_data is not None:
                    self.global_synthetic_data = self.global_synthetic_data.to(self.device)
            
            if "merged_global_graph" in server_msg:
                self.merged_global_graph = server_msg["merged_global_graph"]
                if self.merged_global_graph is not None:
                    self.merged_global_graph = self.merged_global_graph.to(self.device)
            if "server_neigh_gen_cls_weights" in server_msg and server_msg["server_neigh_gen_cls_weights"] is not None:
                gen_weights_list = server_msg["server_neigh_gen_cls_weights"]
                
                for cls_idx in range(self.num_classes):
                    if cls_idx < len(gen_weights_list):
                        weights = gen_weights_list[cls_idx]
                        set_parameters(self.neigh_cls_gen[cls_idx], weights)
           
        if current_round == 0:
            # Pretrain Generator
            self.train_generator()
            # Pretrain Local Model
            self.pretrain()

        # Main Train
        self.train_model(current_round)

    def send_message(self):
        """
        Đóng gói dữ liệu gửi về Server.
        """
        self.upload_weights = (
            get_parameters(self.embedder),
            get_parameters(self.classifier),
            get_parameters(self.clsf_head),
            get_parameters(self.clsf_tail)
        )
        neigh_gen_weights = get_parameters(self.neigh_gen)
        self.message_pool[f"client_{self.client_id}"] = {
            "weight": self.upload_weights,
            "neigh_gen_weights": neigh_gen_weights,
            "num_samples": self.local_graph.x.shape[0],
            "cd_graph": self.upload_cd_graph.cpu() if self.upload_cd_graph else None,
            "cd_scaler": self.cls_cd_scaler.cpu(),
            "train_loss": self.train_loss,
            "train_acc": self.train_acc
        }


    def train_generator(self):
        local_graph_cpu = self.local_graph.clone().cpu()
        
        k_hop = 1
        best_loss = 1e9
        patience = 0
        
        num_nodes = local_graph_cpu.x.shape[0]
        
        batch_list = []
        for i in range(num_nodes):
            sub_idx, sub_edge_index, _, _ = k_hop_subgraph(
                i, k_hop, local_graph_cpu.edge_index, relabel_nodes=True, num_nodes=num_nodes
            )
            sub_x = local_graph_cpu.x[sub_idx]
            batch_list.append(Data(x=sub_x, edge_index=sub_edge_index))
        
        batch_data = Batch.from_data_list(batch_list).to(self.device)

        # Temporary models
        rand_gnn = GraphSAGE(in_channels=self.num_features, hidden_channels=self.args.hid_dim*2, out_channels=self.args.hid_dim, dropout=0.5, num_layers=2).to(self.device)
        rand_classifier = Linear(self.args.hid_dim, self.num_classes).to(self.device)
        
        feats_optim = torch.optim.Adam(self.neigh_gen.parameters(), lr=0.01)
        
        train_mask_part = self.local_graph.train_mask

        true_neigh_feats = []
        for center_id in range(num_nodes):
            neighbors_subset = k_hop_subgraph(
                center_id, num_hops=k_hop, edge_index=local_graph_cpu.edge_index, num_nodes=num_nodes
            )[0]
            
            neighbors = np.setdiff1d(neighbors_subset.numpy(), center_id)
            if len(neighbors) > 0:
                neighbors_proto = local_graph_cpu.x[neighbors].mean(0).unsqueeze(0).numpy()
            else:
                neighbors_proto = np.zeros((1, self.num_features))
            true_neigh_feats.append(neighbors_proto)
            
        true_neigh_feats = torch.Tensor(np.asarray(true_neigh_feats).reshape((-1, self.num_features))).to(self.device)

        pair_edge_index = torch.LongTensor([[0,1],[1,0]]).to(self.device)

        for epoch in range(self.args.pre_gen_epochs):
            self.neigh_gen.train()
            feats_optim.zero_grad()

            input_x = self.local_graph.x
            
            # Forward
            pred_feat = self.neigh_gen(input_x)

            # Synthetic batch construction
            syn_batch_list = []
            for idx in range(pred_feat.shape[0]):
                temp_x = torch.cat([input_x[idx].unsqueeze(0), pred_feat[idx].unsqueeze(0)], 0)
                temp_data = Data(x=temp_x, edge_index=pair_edge_index)
                syn_batch_list.append(temp_data)
            syn_batch_data = Batch.from_data_list(syn_batch_list).to(self.device)

            gen_feat_loss = F.mse_loss(pred_feat[train_mask_part], true_neigh_feats[train_mask_part])

            # Gradient matching logic
            grad_loss = 0
            grad_epochs = 5 
            
            for k in range(grad_epochs):
                rand_gnn.reset_parameters()
                rand_classifier.reset_parameters()
                rand_gnn.train()
                rand_classifier.train()

                # True grads
                true_output = rand_gnn(self.local_graph.x, self.local_graph.edge_index)
                true_pred = rand_classifier(true_output)
                temp_loss = F.cross_entropy(true_pred[train_mask_part], self.local_graph.y[train_mask_part])
                temp_grad = torch.autograd.grad(temp_loss, rand_gnn.parameters(), create_graph=True)
                true_grads = {name: grad for name, grad in zip(rand_gnn.state_dict().keys(), temp_grad)}

                # Syn grads
                gen_output = rand_gnn(syn_batch_data.x, syn_batch_data.edge_index)
                gen_output = gen_output[syn_batch_data.ptr[:-1]]
                gen_pred = rand_classifier(gen_output)
                
                temp_loss = F.cross_entropy(gen_pred[train_mask_part], self.local_graph.y[train_mask_part])
                temp_grad = torch.autograd.grad(temp_loss, rand_gnn.parameters(), create_graph=True)
                gen_grads = {name: grad for name, grad in zip(rand_gnn.state_dict().keys(), temp_grad)}

                for key in gen_grads.keys():
                    if gen_grads[key] is not None and true_grads[key] is not None:
                        grad_loss += F.mse_loss(gen_grads[key], true_grads[key]).sum()
            
            grad_loss /= grad_epochs
            loss = gen_feat_loss + grad_loss
            loss.backward()
            feats_optim.step()

            if loss < best_loss:
                best_loss = loss
                patience = 0
            else:
                patience += 1
            if patience > 20:
                break
        
        # Note: Original code transfers neigh_gen weights to server here. 
        # In this implementation, i keep it local or send if required.

    def pretrain(self):
        optimizer = torch.optim.Adam([
            {'params': self.embedder.parameters()},
            {'params': self.classifier.parameters()}
        ], lr=self.args.lr, weight_decay=5e-4)

        train_mask = self.local_graph.train_mask
        test_mask = self.local_graph.test_mask
        
        best_acc = 0
        patience = 0

        for epoch in range(self.args.pre_epochs):
            self.embedder.train()
            self.classifier.train()
            optimizer.zero_grad()

            h = self.embedder(self.local_graph.x, self.local_graph.edge_index)
            output = self.classifier(h)
            loss = F.cross_entropy(output[train_mask], self.local_graph.y[train_mask])
            loss.backward()
            optimizer.step()

            # Eval
            with torch.no_grad():
                self.embedder.eval()
                self.classifier.eval()
                h = self.embedder(self.local_graph.x, self.local_graph.edge_index)
                output = self.classifier(h)
                acc = accuracy(output[test_mask], self.local_graph.y[test_mask])
                
                if acc >= best_acc:
                    best_acc = acc
                    patience = 0
                else:
                    patience += 1
                
                if patience == 10:
                    break

    def train_model(self, cur_round):
        # synthetic features
        self.syn_feat_head = torch.nn.Parameter(self.syn_feat_head)
        self.syn_feat_tail = torch.nn.Parameter(self.syn_feat_tail)
        
        self.optimizer = torch.optim.Adam([
        {'params': self.embedder.parameters()},
        {'params': self.classifier.parameters()},
        {'params': self.clsf_head.parameters()},
        {'params': self.clsf_tail.parameters()}],
        lr=self.args.lr, weight_decay=5e-4
    )
        loss_fn = torch.nn.NLLLoss()
        
        self.embedder.train()
        self.classifier.train()
        self.clsf_head.train()
        self.clsf_tail.train()
        
        train_mask = self.local_graph.train_mask
        
        epoch_loss = 0.0
        epoch_acc = 0.0

        for epoch in range(self.args.local_epochs):
            self.optimizer.zero_grad()
            self.optim_syn_head.zero_grad()
            self.optim_syn_tail.zero_grad()

            h = self.embedder(self.local_graph.x, self.local_graph.edge_index)
            
            # Synthetic Head/Tail embeddings
            syn_h_head = self.embedder(self.syn_feat_head, self.syn_adj)
            syn_proto_head = syn_h_head.view([self.n_way, self.k_shot, syn_h_head.shape[1]]).mean(1)
            
            syn_h_tail = self.embedder(self.syn_feat_tail, self.syn_adj)
            syn_proto_tail = syn_h_tail.view([self.n_way, self.k_shot, syn_h_tail.shape[1]]).mean(1)

            global_proto = None 
            if cur_round > 0 and self.merged_global_graph is not None:
                    with torch.no_grad():
                        merged_global_graph = self.merged_global_graph
                        
                        global_h = self.embedder(merged_global_graph.x, merged_global_graph.edge_index)
                        global_proto = torch.zeros(self.global_num_classes, h.size(1), dtype=h.dtype, device=self.device)
                        
                        for cls_label in range(self.global_num_classes):
                            class_indices = (merged_global_graph.y == cls_label).nonzero(as_tuple=True)[0]
                            if len(class_indices) == 0: continue
                            num_total_rates = merged_global_graph.cls_rate.size(0)
                            num_clients_in_merge = num_total_rates // self.global_num_classes
                            nodes_per_client = merged_global_graph.x.size(0) // num_clients_in_merge

                            client_indices_of_nodes = class_indices // nodes_per_client
                            rate_indices = client_indices_of_nodes * self.global_num_classes + cls_label                            
                            client_cls_rate = merged_global_graph.cls_rate[rate_indices]

                            if (client_cls_rate == 0).sum() == len(client_cls_rate):
                                client_cls_rate = torch.ones_like(client_cls_rate) / len(client_cls_rate)
                            else:
                                client_cls_rate = client_cls_rate / (client_cls_rate).sum()
                            
                            proto_embedding = (global_h[class_indices] * client_cls_rate.unsqueeze(1)).sum(0)
                            global_proto[cls_label] = proto_embedding

            h_train = h[train_mask]
            y_train = self.local_graph.y[train_mask]
            
            edge1 = []
            current_batch_size = len(h[train_mask]) 
                
            start_node_idx = self.n_way * self.k_shot
            end_node_idx = start_node_idx + current_batch_size

            for i in range(start_node_idx, end_node_idx):
                    temp = [i] * (self.n_way * self.k_shot)
                    edge1.extend(temp)
                    
            edge1 = torch.LongTensor(edge1)
            edge2 = torch.LongTensor(list(range(self.n_way * self.k_shot)) * current_batch_size)
                
            metric_edge_index = torch.stack((edge2, edge1)).to(self.device)

            edge_index_wo_self = remove_self_loops(self.local_graph.edge_index)[0]
            adj_matrix = SparseTensor(row=edge_index_wo_self[0], col=edge_index_wo_self[1], 
                                        sparse_sizes=(h.size(0), h.size(0)))
            neighbor = adj_matrix @ h
            norm = adj_matrix.sum(0)
            norm[norm == 0] = 1
            neighbor = neighbor / norm.unsqueeze(1)
                
            neighbor_train = neighbor[train_mask] 
            _, embeds_epi_head = self.clsf_head(h[train_mask], neighbor_train, syn_h_head, metric_edge_index)
            _, embeds_epi_tail = self.clsf_tail(h[train_mask], neighbor_train, syn_h_tail, metric_edge_index)

            # Prediction & Loss
            dists_head = euclidean_dist(embeds_epi_head, syn_proto_head)
            dists_tail = euclidean_dist(embeds_epi_tail, syn_proto_tail)
            
            out_head = F.log_softmax(-dists_head, dim=1)
            out_tail = F.log_softmax(-dists_tail, dim=1)
            
            # Alpha gating
            node_deg = degree(self.local_graph.edge_index[0], num_nodes=self.local_graph.x.shape[0])
            alpha = 1 / (1 + torch.exp(-(node_deg - (self.args.head_deg_thres + 1))))
            alpha = alpha[train_mask].unsqueeze(1).to(self.device)
            
            output_metric = alpha * out_head + (1 - alpha) * out_tail
            loss_metric = loss_fn(output_metric, y_train)
            
            syn_norm_loss = 0.5 * torch.mean(torch.norm(self.syn_feat_head)) + \
                            0.5 * torch.mean(torch.norm(self.syn_feat_tail))

            # Cross Domain Loss (with Global Synthetic Data)
            loss_cd_metric = 0
            loss_cd_metric = 0
            if cur_round > 0 and self.global_synthetic_data is not None:
                    g_syn_data = self.global_synthetic_data
                    syn_xs = g_syn_data.x.to(self.device) # Shape [N, F]
                    syn_ys = g_syn_data.y.to(self.device)
                    
                    # 1. Normalize & Scale Synthetic Data
                    mean_syn = syn_xs.mean(0).unsqueeze(0)
                    cls_scale = torch.zeros_like(syn_ys).float().to(self.device)
                    for c in range(len(self.cls_cd_scaler)):
                        cls_scale[syn_ys == c] = self.cls_cd_scaler[c]
                    
                    weighted_global_syn = syn_xs + cls_scale.unsqueeze(1) * (mean_syn - syn_xs)
                    
                    # 2. Generate Neighbors cho Synthetic Data 
                    syn_graph_list = []
                    with torch.no_grad():
                        for cls_idx in range(self.num_classes):
                            neigh_gen_cls = self.neigh_cls_gen[cls_idx]
                            
                            cls_mask = (syn_ys == cls_idx)
                            if cls_mask.sum() == 0: continue
                            
                            cls_inputs = weighted_global_syn[cls_mask]
                            gen_feats = neigh_gen_cls(cls_inputs)
                            
                            for k in range(gen_feats.shape[0]):
                                neigh_x = torch.cat([cls_inputs[k].unsqueeze(0), gen_feats[k].unsqueeze(0)], 0)
                                neigh_edges = torch.LongTensor([[0,1],[1,0]]).to(self.device)
                                syn_graph_list.append(Data(x=neigh_x, edge_index=neigh_edges))
                    
                    if len(syn_graph_list) > 0:
                        syn_batch = Batch.from_data_list(syn_graph_list).to(self.device)
                        
                        global_h_all = self.embedder(syn_batch.x, syn_batch.edge_index)
                        global_h = global_h_all[syn_batch.ptr[:-1]] 
                        n_syn = len(global_h)
                        edge1_cd = []
                        start_idx = self.n_way * self.k_shot
                        for i in range(start_idx, start_idx + n_syn):
                            edge1_cd.extend([i] * (self.n_way * self.k_shot))
                        
                        edge1_cd = torch.LongTensor(edge1_cd)
                        edge2_cd = torch.LongTensor(list(range(self.n_way * self.k_shot)) * n_syn)
                        cd_edge_index = torch.stack((edge2_cd, edge1_cd)).to(self.device)
                        
                        with torch.no_grad():
                            edge_index_wo_syn = remove_self_loops(syn_batch.edge_index)[0]
                            adj_syn = SparseTensor(row=edge_index_wo_syn[0], col=edge_index_wo_syn[1],
                                                 sparse_sizes=(global_h_all.size(0), global_h_all.size(0)))
                            neighbor_syn = adj_syn @ global_h_all
                            norm_syn = adj_syn.sum(0)
                            norm_syn[norm_syn == 0] = 1
                            neighbor_syn = neighbor_syn / norm_syn.unsqueeze(1)
                            neighbor_syn = neighbor_syn[syn_batch.ptr[:-1]]

                        _, embeds_global_head = self.clsf_head(global_h, neighbor_syn, syn_h_head, cd_edge_index)
                        _, embeds_global_tail = self.clsf_tail(global_h, neighbor_syn, syn_h_tail, cd_edge_index)
                        
                        dists_cd_head = euclidean_dist(embeds_global_head, syn_proto_head)
                        dists_cd_tail = euclidean_dist(embeds_global_tail, syn_proto_tail)
                        
                        out_cd_head = F.log_softmax(-dists_cd_head, dim=1)
                        out_cd_tail = F.log_softmax(-dists_cd_tail, dim=1)
                        
                        out_cd_metric = 0.5 * out_cd_head + 0.5 * out_cd_tail

                        batch_syn_y = []
                        for c in range(self.num_classes):
                            count = (syn_ys == c).sum().item()
                            if count > 0:
                                batch_syn_y.extend([c] * count)
                        batch_syn_y = torch.LongTensor(batch_syn_y).to(self.device)

                        loss_cd_metric = loss_fn(out_cd_metric, batch_syn_y)
                        with torch.no_grad():
                            preds = torch.argmax(out_cd_metric, dim=1)
                            for c in range(self.num_classes):
                                mask_c = (batch_syn_y == c)
                                if mask_c.sum() == 0: continue
                                
                                correct = (preds[mask_c] == c).float()
                                acc_c = correct.mean().item()
                                
                                if acc_c > 0.8:
                                    self.cls_cd_scaler[c] += 0.001
                                else:
                                    self.cls_cd_scaler[c] -= 0.001
                            
                            self.cls_cd_scaler.clamp_(0.0, 1.0)

            if cur_round > 0:
                loss = self.args.hyper_metric * loss_metric + \
                       self.args.hyper_syn_norm * syn_norm_loss + \
                       self.args.hyper_cd_metric * loss_cd_metric
            else:
                loss = self.args.hyper_metric * loss_metric + \
                       self.args.hyper_syn_norm * syn_norm_loss

            loss.backward()
            self.optimizer.step()
            self.optim_syn_head.step()
            self.optim_syn_tail.step()
            
            epoch_loss += loss.item()
            acc = accuracy(output_metric.detach(), y_train)
            epoch_acc += acc

        self.train_loss = epoch_loss / self.args.local_epochs
        self.train_acc = epoch_acc / self.args.local_epochs
        cond_graph = Data(edge_index=to_edge_index(self.syn_adj)[0].cpu(), y=self.syn_y.cpu())
        
        cond_graph.x_head = self.syn_feat_head.clone().detach().cpu()
        cond_graph.x_tail = self.syn_feat_tail.clone().detach().cpu()
        
        x_head_proto = cond_graph.x_head.reshape([self.n_way, self.k_shot, -1]).mean(1)
        cond_graph.x_head_proto = x_head_proto
        
        x_tail_proto = cond_graph.x_tail.reshape([self.n_way, self.k_shot, -1]).mean(1)
        cond_graph.x_tail_proto = x_tail_proto
        
        cls_count = torch.bincount(self.local_graph.y[train_mask].cpu(), minlength=self.n_way)
        cond_graph.cls_rate = (cls_count / cls_count.sum()).clone().detach().float() # .float() để tránh lỗi chia int
        
        self.upload_cd_graph = cond_graph
        
def test(self, data=None):
        if data is None:
            data = self.local_graph
        
        data = data.to(self.device)
        
        self.embedder.eval()
        self.clsf_head.eval()
        self.clsf_tail.eval()
        self.classifier.eval() 
        with torch.no_grad():
            h = self.embedder(data.x, data.edge_index)
            

            syn_h_head = self.embedder(self.syn_feat_head, self.syn_adj)
            syn_proto_head = syn_h_head.view([self.n_way, self.k_shot, syn_h_head.shape[1]]).mean(1)

            syn_h_tail = self.embedder(self.syn_feat_tail, self.syn_adj)
            syn_proto_tail = syn_h_tail.view([self.n_way, self.k_shot, syn_h_tail.shape[1]]).mean(1)

            num_nodes = h.size(0)
            num_protos = self.n_way * self.k_shot

            edge_index_wo_self = remove_self_loops(data.edge_index)[0]
            adj_matrix = SparseTensor(row=edge_index_wo_self[0], col=edge_index_wo_self[1], 
                                      sparse_sizes=(num_nodes, num_nodes))
            neighbor = adj_matrix @ h
            norm = adj_matrix.sum(0)
            norm[norm == 0] = 1
            neighbor = neighbor / norm.unsqueeze(1)

            edge1 = []
            for i in range(num_protos, num_protos + num_nodes):
                edge1.extend([i] * num_protos)
            edge1 = torch.LongTensor(edge1)
            edge2 = torch.LongTensor(list(range(num_protos)) * num_nodes)
            test_edge_index = torch.stack((edge2, edge1)).to(self.device)


            neighbor, embeds_epi_head = self.clsf_head(h, neighbor, syn_h_head, test_edge_index)
            neighbor, embeds_epi_tail = self.clsf_tail(h, neighbor, syn_h_tail, test_edge_index)

            dists_head = euclidean_dist(embeds_epi_head, syn_proto_head)
            dists_tail = euclidean_dist(embeds_epi_tail, syn_proto_tail)

            out_head_soft = F.softmax(-dists_head, dim=1)
            out_tail_soft = F.softmax(-dists_tail, dim=1)

            deg = torch.Tensor(degree(data.edge_index[0], num_nodes=num_nodes).cpu().tolist()).to(self.device)
            alpha = 1 / (1 + torch.exp(-(deg - (self.args.head_deg_thres + 1))))
            alpha = alpha.unsqueeze(1)

            output_final = alpha * out_head_soft + (1 - alpha) * out_tail_soft

            if hasattr(data, 'test_mask') and data.test_mask is not None:
                mask = data.test_mask
                acc = accuracy(output_final[mask], data.y[mask])
                return acc, 0.0 
            return 0.0, 0.0