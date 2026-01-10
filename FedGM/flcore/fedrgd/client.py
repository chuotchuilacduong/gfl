import torch
import copy
import torch.nn.functional as F
from flcore.fedgm.client import FedGMClient
from flcore.fedgm.utils import match_loss, normalize_adj_tensor
from flcore.fedgm.pge import PGE
from flcore.fedgm.IGNR import GraphonLearner as IGNR
from flcore.fedgm.fedgm_config import config
from torch_geometric.utils import to_torch_sparse_tensor, dense_to_sparse
from model.gcn import GCN_kipf
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from types import SimpleNamespace
from utils.metrics import compute_supervised_metrics

def normalize_sparse_gcn(adj_t):
    adj_t = fill_diag(adj_t, 1.0) 
    deg = sparsesum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1)) # D^-0.5 * (A+I)
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1)) # (D^-0.5 * (A+I)) * D^-0.5
    return adj_t


def robust_normalize_adj(adj, eps=0): 

    adj = torch.clamp(adj, min=0, max=10)
    
    adj = adj + torch.eye(adj.shape[0], device=adj.device)
    
    row_sum = torch.sum(adj, 1) + eps
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt = torch.clamp(d_inv_sqrt, min=0, max=10) 
    d_inv_sqrt = d_inv_sqrt.view(-1, 1) 
    adj_norm = adj * d_inv_sqrt * d_inv_sqrt.view(1, -1)
    
    return adj_norm
def sample_edges_from_probabilistic_adj(adj, threshold=0.3, top_k=15, device=None):
    if device is None:
        device = adj.device
    
    num_nodes = adj.size(0)
    if num_nodes == 0:
        return (torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.float, device=device))

    top_k_actual = min(top_k, num_nodes) if top_k is not None else num_nodes
    
    vals, indices = torch.topk(adj, top_k_actual, dim=1, largest=True)
    
    mask = vals > threshold
    
    row_indices = torch.arange(num_nodes, device=device).unsqueeze(1).expand_as(indices)
    src = row_indices[mask]
    dst = indices[mask]   
    if src.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_weight = torch.empty(0, dtype=torch.float, device=device)
    else:
        edge_index = torch.stack([src, dst], dim=0)
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float, device=device)
        
    return edge_index, edge_weight


class FedRGDClient(FedGMClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedRGDClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        if not hasattr(args, 'method'):
            args.method = config.get('method', 'GCond')
        task_model = GCN_kipf(
            nfeat=self.task.num_feats,
            nhid=args.hid_dim,
            nclass=self.task.num_global_classes,
            nlayers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device
        ).to(device)
        self.task.load_custom_model(task_model)
        
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
        
        self.task.override_evaluate = self.get_override_evaluate()
        
        self.local_metrics = {}
        
    def execute(self):
        server_msg = self.message_pool["server"]
        global_graph = server_msg.get("global_graph")

        # Load local data
        local_data = self.task.splitted_data['data']
        l_x, l_y = local_data.x, local_data.y
        l_edge_index = local_data.edge_index
        l_edge_weight = torch.ones(l_edge_index.size(1), device=self.device)
        l_mask = self.task.splitted_data['train_mask']

        if global_graph is not None:
            g_x = global_graph['x'].to(self.device).detach()
            g_y = global_graph['y'].to(self.device)
            g_adj = global_graph['adj'].to(self.device).detach()
            
            g_edge_index, g_edge_weight = sample_edges_from_probabilistic_adj(
                g_adj,
                threshold=0.2,
                top_k=15,
                device=self.device
            )
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
            row=batch_edge_index[0], 
            col=batch_edge_index[1], 
            value=batch_edge_weight,
            sparse_sizes=(batch_x.size(0), batch_x.size(0))
        ).to(self.device)
        
        batch_adj_norm = normalize_sparse_gcn(batch_adj_sparse)
        
        if server_msg.get("weights"):
            with torch.no_grad():
                for (local_param, global_param) in zip(self.task.model.parameters(), server_msg["weights"]):
                    local_param.data.copy_(global_param)
        
        # Train Classification Model
        self.task.model.train()
        optimizer_cls = self.task.optim 
        
        for epoch in range(self.args.local_epochs):
            optimizer_cls.zero_grad()
            
            output = self.task.model(batch_x, batch_adj_norm)
            
            loss_cls = F.nll_loss(output[batch_mask], batch_y[batch_mask])
            
            loss_cls.backward()
            torch.nn.utils.clip_grad_norm_(self.task.model.parameters(), max_norm=1.0)
            optimizer_cls.step()

        # Evaluate
        local_eval_results = self.task.evaluate(splitted_data=self.task.splitted_data, mute=True)
        self.local_metrics = {
            'loss_train': local_eval_results.get('loss_train', torch.tensor(float('nan'))).item(),
            'accuracy_train': local_eval_results.get('accuracy_train', float('nan')),
            'loss_val': local_eval_results.get('loss_val', torch.tensor(float('nan'))).item(),
            'accuracy_val': local_eval_results.get('accuracy_val', float('nan')),
            'loss_test': local_eval_results.get('loss_test', torch.tensor(float('nan'))).item(),
            'accuracy_test': local_eval_results.get('accuracy_test', float('nan'))
        }
        
        # Graph Condensation
        if self.message_pool["round"] > 0:
            batch_adj_dense = batch_adj_sparse.to_dense()
            self._perform_graph_condensation(batch_x, batch_y, batch_adj_dense, batch_mask)
    
    def _perform_graph_condensation(self, batch_x, batch_y, batch_adj_dense, batch_mask):

        self.model_cond.load_state_dict(self.task.model.state_dict())
        
        self.model_cond.eval() 
        for p in self.model_cond.parameters():
            p.requires_grad = False 
        batch_adj_norm = robust_normalize_adj(batch_adj_dense)
        syn_class_indices = {}
        for c in range(self.task.num_global_classes):
            indices = (self.syn_y == c).nonzero(as_tuple=True)[0]
            if len(indices) > 0:
                syn_class_indices[c] = indices

        real_gradients_cache = [] 
        for p in self.model_cond.parameters():
            p.requires_grad = True

        self.model_cond.zero_grad()
        output_real = self.model_cond(batch_x, batch_adj_norm)
        unique_classes = torch.unique(batch_y[batch_mask])
        
        for c in unique_classes:
            c = c.item()
            if c not in syn_class_indices:
                continue

            mask_c = batch_mask & (batch_y == c)
            if mask_c.sum() == 0:
                continue
            
            loss_real_c = F.nll_loss(output_real[mask_c], batch_y[mask_c])
            
            gw_real_c = torch.autograd.grad(
                loss_real_c, 
                self.model_cond.parameters(), 
                retain_graph=True, 
                create_graph=False 
            )
            gw_real_c = [g.detach() for g in gw_real_c]
            real_gradients_cache.append((c, gw_real_c))
            
        del output_real, loss_real_c
        torch.cuda.empty_cache() 
        self.model_cond.train()
        for it in range(self.args.condense_iters):
            self.optimizer_feat.zero_grad()
            self.optimizer_pge.zero_grad()

            # Forward Synthetic
            if self.args.method == 'SGDD':
                 adj_syn, opt_loss = self.pge(self.syn_x, Lx=None)
            else:
                 adj_syn = self.pge(self.syn_x)
                 opt_loss = torch.tensor(0.0, device=self.device)
            adj_syn_norm = robust_normalize_adj(adj_syn)
            output_syn = self.model_cond(self.syn_x, adj_syn_norm)
            
            total_match_loss = 0.0
            
            for c, gw_real_c in real_gradients_cache:
                idx_c = syn_class_indices[c]
                
                loss_syn_c = F.nll_loss(output_syn[idx_c], self.syn_y[idx_c])
                
                gw_syn_c = torch.autograd.grad(
                    loss_syn_c, 
                    self.model_cond.parameters(), 
                    create_graph=True, 
                    retain_graph=True
                )                
                match_loss_c = match_loss(
                    gw_syn_c, gw_real_c, 
                    dis_metric=config["dis_metric"], 
                    device=self.device
                )
                total_match_loss += match_loss_c
            if self.args.method == 'SGDD' and config.get("opt_scale", 0) > 0:
                 total_match_loss += config["opt_scale"] * opt_loss
            # Update
            if total_match_loss > 0:
                total_match_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.pge.parameters()) + [self.syn_x],
                    max_norm=1.0
                )
                self.optimizer_feat.step()
                self.optimizer_pge.step()
    
    def get_override_evaluate(self):
        
        def override_evaluate(splitted_data=None, mute=False):
            if splitted_data is None:
                splitted_data = self.task.splitted_data
            
            data = splitted_data['data']
            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            y = data.y.to(self.device)
            
            adj_sparse = SparseTensor(
                row=edge_index[0], col=edge_index[1], 
                sparse_sizes=(x.size(0), x.size(0))
            ).to(self.device)
            adj_norm = normalize_sparse_gcn(adj_sparse)
            
            self.task.model.eval()
            with torch.no_grad():
                # GCN_kipf forward: (x, adj)
                logits = self.task.model(x, adj_norm)
                
                loss_train = F.nll_loss(logits[splitted_data['train_mask']], y[splitted_data['train_mask']])
                loss_val = F.nll_loss(logits[splitted_data['val_mask']], y[splitted_data['val_mask']])
                loss_test = F.nll_loss(logits[splitted_data['test_mask']], y[splitted_data['test_mask']])

            eval_output = {
                "logits": logits,
                "loss_train": loss_train,
                "loss_val": loss_val,
                "loss_test": loss_test,
                "embedding": None
            }
            
            metric_train = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["train_mask"]], labels=y[splitted_data["train_mask"]], suffix="train")
            metric_val = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["val_mask"]], labels=y[splitted_data["val_mask"]], suffix="val")
            metric_test = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["test_mask"]], labels=y[splitted_data["test_mask"]], suffix="test")
            
            eval_output.update(metric_train)
            eval_output.update(metric_val)
            eval_output.update(metric_test)
            
            if not mute:
                print(f"[Client {self.client_id}] Eval: Train {metric_train}, Val {metric_val}, Test {metric_test}")
                
            return eval_output
            
        return override_evaluate

    def send_message(self):
        with torch.no_grad():
            if self.args.method == 'SGDD':
                final_adj, _ = self.pge(self.syn_x, Lx=None)
            else:
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