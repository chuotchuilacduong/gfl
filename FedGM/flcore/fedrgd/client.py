import torch
import copy
import torch.nn.functional as F
from flcore.fedgm.client import FedGMClient
from flcore.fedgm.utils import match_loss, normalize_adj_tensor
from flcore.fedgm.pge import PGE
from flcore.fedgm.fedgm_config import config
from torch_geometric.utils import to_torch_sparse_tensor,dense_to_sparse
from model.gcn import GCN_kipf
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from types import SimpleNamespace

def normalize_sparse_gcn(adj_t):

    adj_t = fill_diag(adj_t, 1.0)
    deg = sparsesum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t


class FedRGDClient(FedGMClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedRGDClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.task.override_evaluate = None
        self.model_cond = GCN_kipf(
            nfeat=self.task.num_feats,
            nhid=args.hid_dim,
            nclass=self.task.num_global_classes,
            nlayers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device
        ).to(device)
        self.local_metrics = {}
        
    def process_global_graph_for_kipf(global_data, device, threshold=0.01):
        x = global_data['x'].to(device)
        y = global_data['y'].to(device)
        adj = global_data['adj'].to(device) # Dense (N, N)

        if threshold > 0:
            mask = adj > threshold
            adj = adj * mask

        edge_index, edge_weight = dense_to_sparse(adj)
        
        adj_sparse = SparseTensor(
            row=edge_index[0], 
            col=edge_index[1], 
            value=edge_weight,
            sparse_sizes=(x.size(0), x.size(0))
        ).to(device)

        return x, y, adj_sparse
    def execute(self):
        server_msg = self.message_pool["server"]
        global_graph = server_msg.get("global_graph")

        local_data = self.task.splitted_data['data']
        l_x, l_y = local_data.x, local_data.y
        l_edge_index = local_data.edge_index
        l_edge_weight = torch.ones(l_edge_index.size(1), device=self.device) # Weight = 1.0 cho đồ thị thật
        l_mask = self.task.splitted_data['train_mask']

        if global_graph is not None:
            g_x = global_graph['x'].to(self.device)
            g_y = global_graph['y'].to(self.device)
            g_adj = global_graph['adj'].to(self.device)
            
            mask_threshold = 0.01
            g_adj = g_adj * (g_adj > mask_threshold)
            g_edge_index, g_edge_weight = dense_to_sparse(g_adj)

            batch_x = torch.cat([l_x, g_x], dim=0)
            batch_y = torch.cat([l_y, g_y], dim=0)
            num_local_nodes = l_x.size(0)
            shifted_g_edge_index = g_edge_index + num_local_nodes
            
            batch_edge_index = torch.cat([l_edge_index, shifted_g_edge_index], dim=1)
            batch_edge_weight = torch.cat([l_edge_weight, g_edge_weight], dim=0)
            g_mask = torch.ones(g_x.size(0), dtype=torch.bool, device=self.device)
            batch_mask = torch.cat([l_mask, g_mask], dim=0)

        else:
            batch_x, batch_y = l_x, l_y
            batch_edge_index, batch_edge_weight = l_edge_index, l_edge_weight
            batch_mask = l_mask

        batch_adj_sparse = SparseTensor(
            row=batch_edge_index[0], col=batch_edge_index[1], value=batch_edge_weight,
            sparse_sizes=(batch_x.size(0), batch_x.size(0))
        ).to(self.device)
        
        batch_adj_norm = normalize_sparse_gcn(batch_adj_sparse.clone())
        if server_msg.get("weights"):
            with torch.no_grad():
                for (local_param, global_param) in zip(self.task.model.parameters(), server_msg["weights"]):
                    local_param.data.copy_(global_param)
        
        self.task.model.train()
        optimizer_cls = self.task.optim
        
        for epoch in range(self.args.local_epochs):
            optimizer_cls.zero_grad()
            batch_data = SimpleNamespace(x=batch_x, edge_index=batch_adj_sparse)
            _, output = self.task.model(batch_data)
            
            loss_cls = F.nll_loss(output[batch_mask], batch_y[batch_mask])
            
            loss_cls.backward()
            optimizer_cls.step()

        local_eval_results = self.task.evaluate(splitted_data=self.task.splitted_data, mute=True)
        self.local_metrics = {
            'loss_train': local_eval_results.get('loss_train', torch.tensor(float('nan'))).item(),
            'accuracy_train': local_eval_results.get('accuracy_train', float('nan')),
            'loss_val': local_eval_results.get('loss_val', torch.tensor(float('nan'))).item(),
            'accuracy_val': local_eval_results.get('accuracy_val', float('nan')),
            'loss_test': local_eval_results.get('loss_test', torch.tensor(float('nan'))).item(),
            'accuracy_test': local_eval_results.get('accuracy_test', float('nan'))
        }
        if self.message_pool["round"] > 0:
            
            
            self.model_cond.train()

            for p in self.model_cond.parameters():
                p.requires_grad = True

            syn_class_indices = {}
            for c in range(self.task.num_global_classes):
                indices = (self.syn_y == c).nonzero(as_tuple=True)[0]
                if len(indices) > 0:
                    syn_class_indices[c] = indices

            for it in range(self.args.condense_iters):
                self.model_cond.initialize() 
                self.model_cond.zero_grad()

                # Real/Global Batch
                output_real = self.model_cond(batch_x, batch_adj_norm)
                
                # Synthetic Batch
                adj_syn = self.pge(self.syn_x)
                adj_syn_norm = normalize_adj_tensor(adj_syn, sparse=False)
                output_syn = self.model_cond(self.syn_x, adj_syn_norm)

                total_match_loss = 0.0
                
                unique_classes = torch.unique(batch_y[batch_mask])
                
                for c in unique_classes:
                    c = c.item()
                    if c not in syn_class_indices: continue

                    mask_c = batch_mask & (batch_y == c)
                    if mask_c.sum() == 0: continue
                    
                    loss_real_c = F.nll_loss(output_real[mask_c], batch_y[mask_c])
                    gw_real_c = torch.autograd.grad(loss_real_c, self.model_cond.parameters(), retain_graph=True)
                    gw_real_c = list((_.detach().clone() for _ in gw_real_c))

                    
                    idx_c = syn_class_indices[c]
                    
                    loss_syn_c = F.nll_loss(output_syn[idx_c], self.syn_y[idx_c])
                    gw_syn_c = torch.autograd.grad(loss_syn_c, self.model_cond.parameters(), create_graph=True, retain_graph=True)

                    match_loss_c = match_loss(gw_syn_c, gw_real_c, dis_metric=config["dis_metric"], device=self.device)
                    total_match_loss += match_loss_c

                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                total_match_loss.backward()
                self.optimizer_feat.step()
                self.optimizer_pge.step()
                

    def send_message(self):
        with torch.no_grad():
            final_adj = self.pge.inference(self.syn_x)
        
        self.message_pool[f"client_{self.client_id}"] = {
            "weights": list(self.task.model.parameters()),
            "num_samples": self.task.num_samples,
            "local_metrics": self.local_metrics,
            "syn_graph": {
                "x": self.syn_x.detach().cpu(),
                "adj": final_adj.detach().cpu(), 
                "y": self.syn_y.detach().cpu()
            }
        }