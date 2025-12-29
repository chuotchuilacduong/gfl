import torch
import copy
import torch.nn.functional as F
from flcore.fedgm.server import FedGMServer
from flcore.fedgm.utils import match_loss, normalize_adj_tensor
from flcore.fedgm.pge import PGE
from flcore.fedgm.fedgm_config import config
from model.gcn import GCN_kipf
from torch_sparse import SparseTensor, fill_diag, sum as sparsesum, mul
from utils.metrics import compute_supervised_metrics

def normalize_sparse_gcn(adj_t):

    adj_t = fill_diag(adj_t, 1.0) # A + I
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
    # d_inv_sqrt = torch.clamp(d_inv_sqrt, min=0, max=10)
    # d_inv_sqrt[torch.isnan(d_inv_sqrt) | torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    adj_norm = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt) 
    # if torch.isnan(adj_norm).any() or torch.isinf(adj_norm).any():
    #     adj_norm = torch.nan_to_num(adj_norm, nan=0.0, posinf=1.0, neginf=0.0)
    
    return adj_norm

class FedRGDServer(FedGMServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedRGDServer, self).__init__(args, global_data, data_dir, message_pool, device)
        
        self.num_global_syn_nodes = args.num_global_syn_nodes
        self.pge_global = PGE(nfeat=self.task.num_feats, nnodes=self.num_global_syn_nodes, device=device, args=args).to(device)
        self.syn_x_global = torch.randn(self.num_global_syn_nodes, self.task.num_feats, requires_grad=True, device=device)
        self.syn_y_global = torch.LongTensor([i % self.task.num_global_classes for i in range(self.num_global_syn_nodes)]).to(device)
        
        self.optimizer_global = torch.optim.Adam([
            {'params': [self.syn_x_global], 'lr': config['lr_feat']},
            {'params': self.pge_global.parameters(), 'lr': config['lr_adj']}
        ])
        
        # 1. Classification Model (task.model): GCN_kipf
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

        # 2. Condensation Model (model_cond): GCN_kipf
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
        
        self.global_graph_cache = None
        self.last_gradient_match_loss = float('nan')
        
        # print(f"[Server] Initialized. Using GCN_kipf for both Task and Condensation.")

    def aggregate_classification_weights(self):
        with torch.no_grad():
            valid_clients = [cid for cid in self.message_pool["sampled_clients"] 
                             if f"client_{cid}" in self.message_pool and "weights" in self.message_pool[f"client_{cid}"]]
            
            if not valid_clients:
                return

            num_tot_samples = sum([self.message_pool[f"client_{cid}"]["num_samples"] for cid in valid_clients])
            
            for it, client_id in enumerate(valid_clients):
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                client_weights = self.message_pool[f"client_{client_id}"]["weights"]
                
                for (local_param, global_param) in zip(client_weights, self.task.model.parameters()):
                    if it == 0:
                        global_param.data.copy_(weight * local_param)
                    else:
                        global_param.data += weight * local_param
    
    def execute(self):
        """Main server execution: aggregate weights and perform graph condensation"""
        self.aggregate_classification_weights()
        self.last_gradient_match_loss = float('nan')
        
        if self.message_pool["round"] > 0:
            self._perform_graph_condensation()
            
        self._cache_global_graph()
    
    def _perform_graph_condensation(self):
        """
        gradient matching-based graph condensation
        """
        self.model_cond.load_state_dict(self.task.model.state_dict())
        
        for p in self.model_cond.parameters():
            p.requires_grad = True
        
        global_syn_class_indices = {}
        for c in range(self.task.num_global_classes):
            indices = (self.syn_y_global == c).nonzero(as_tuple=True)[0]
            if len(indices) > 0:
                global_syn_class_indices[c] = indices
        
        input_graphs = []
        for cid in self.message_pool["sampled_clients"]:
            msg = self.message_pool.get(f"client_{cid}")
            if msg and "syn_graph" in msg:
                input_graphs.append(msg["syn_graph"])

        if not input_graphs:
            return
        target_cache = []
        
        self.model_cond.eval() 
        self.model_cond.zero_grad()

        for client_graph in input_graphs:
            c_x = client_graph["x"].to(self.device).detach()
            c_y = client_graph["y"].to(self.device)
            c_adj_dense = client_graph["adj"].to(self.device).detach()

            if torch.isnan(c_x).any() or torch.isnan(c_adj_dense).any():
                continue

            c_adj_norm = robust_normalize_adj(c_adj_dense)
            
            out_target = self.model_cond(c_x, c_adj_norm)
            
            if not self._is_log_softmax(out_target):
                out_target = torch.nn.functional.log_softmax(out_target, dim=1)
            
            loss_target = torch.nn.functional.nll_loss(out_target, c_y)
            
            if torch.isnan(loss_target) or torch.isinf(loss_target):
                continue
            
            gw_target = torch.autograd.grad(
                loss_target, 
                self.model_cond.parameters(),
                retain_graph=False, 
                create_graph=False
            )
            gw_target = [g.detach() for g in gw_target] 

            unique_classes_client = torch.unique(c_y)
            relevant_syn_indices = []
            for c in unique_classes_client:
                c_item = c.item()
                if c_item in global_syn_class_indices:
                    relevant_syn_indices.append(global_syn_class_indices[c_item])
            
            if relevant_syn_indices:
                batch_syn_indices = torch.cat(relevant_syn_indices)
                target_cache.append({
                    'gw_target': gw_target,
                    'syn_indices': batch_syn_indices
                })
        
        if not target_cache:
            return

        self.model_cond.train() 
        for it in range(self.args.server_condense_iters):
            self.optimizer_global.zero_grad()
            self.model_cond.zero_grad()
            
            # 1. Forward Synthetic Global
            adj_syn_global = self.pge_global(self.syn_x_global)
            if torch.isnan(adj_syn_global).any() or torch.isinf(adj_syn_global).any():
                continue
            
            adj_syn_norm = robust_normalize_adj(adj_syn_global)
            out_syn_global = self.model_cond(self.syn_x_global, adj_syn_norm)
            
            if not self._is_log_softmax(out_syn_global):
                out_syn_global = torch.nn.functional.log_softmax(out_syn_global, dim=1)
            
            # 2. Match Gradients
            total_match_loss = 0.0
            valid_graphs = 0
            
            for item in target_cache:
                gw_target = item['gw_target']
                batch_syn_indices = item['syn_indices']
                
                loss_syn = torch.nn.functional.nll_loss(
                    out_syn_global[batch_syn_indices],
                    self.syn_y_global[batch_syn_indices]
                )
                
                if torch.isnan(loss_syn) or torch.isinf(loss_syn):
                    continue

                # Gradient Synthetic (create_graph=True)
                gw_syn = torch.autograd.grad(
                    loss_syn, 
                    self.model_cond.parameters(), 
                    retain_graph=True,
                    create_graph=True
                )
                
                batch_loss = match_loss(gw_syn, gw_target, dis_metric=config["dis_metric"], device=self.device)
                total_match_loss += batch_loss
                valid_graphs += 1
            
            if valid_graphs == 0:
                continue
            
            avg_loss = total_match_loss / valid_graphs
            avg_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                list(self.pge_global.parameters()) + [self.syn_x_global],
                max_norm=1.0
            )
            
            self.optimizer_global.step()
            self.last_gradient_match_loss = avg_loss.item()
            
            del out_syn_global, adj_syn_norm, adj_syn_global
            # torch.cuda.empty_cache()
        
        print(f"{'='*60}\n")
    
    def _compute_gradients_for_client(self, client_graph, out_syn_global, global_syn_class_indices):
        """Compute gradients for a single client graph."""
        c_x = client_graph["x"].to(self.device).detach()
        c_y = client_graph["y"].to(self.device)
        c_adj_dense = client_graph["adj"].to(self.device).detach()

        if torch.isnan(c_x).any() or torch.isnan(c_adj_dense).any():
            return None

        c_adj_norm = robust_normalize_adj(c_adj_dense)
        
        out_target = self.model_cond(c_x, c_adj_norm)
        
        if not self._is_log_softmax(out_target):
            out_target = torch.nn.functional.log_softmax(out_target, dim=1)
        
        loss_target = torch.nn.functional.nll_loss(out_target, c_y)
        
        if torch.isnan(loss_target) or torch.isinf(loss_target):
            return None
        
        # Gradient Target (Real) - create_graph=False
        gw_target = torch.autograd.grad(
            loss_target, 
            self.model_cond.parameters(),
            retain_graph=True,
            create_graph=False
        )
        gw_target = [g.detach().clone() for g in gw_target]
        
        unique_classes_client = torch.unique(c_y)
        relevant_syn_indices = []
        for c in unique_classes_client:
            c_item = c.item()
            if c_item in global_syn_class_indices:
                relevant_syn_indices.append(global_syn_class_indices[c_item])
        
        if not relevant_syn_indices:
            return None
        
        batch_syn_indices = torch.cat(relevant_syn_indices)
        
        loss_syn = torch.nn.functional.nll_loss(
            out_syn_global[batch_syn_indices],
            self.syn_y_global[batch_syn_indices]
        )
        
        if torch.isnan(loss_syn) or torch.isinf(loss_syn):
            return None
        
        # Gradient Synthetic - create_graph=True
        gw_syn = torch.autograd.grad(
            loss_syn, 
            self.model_cond.parameters(), 
            retain_graph=True,
            create_graph=True
        )
        
        return (gw_syn, gw_target)
    
    def _is_log_softmax(self, output, atol=1e-3):
        exp_sum = torch.exp(output).sum(dim=1)
        expected = torch.ones(output.size(0), device=self.device)
        return torch.allclose(exp_sum, expected, atol=atol)
    
    def _cache_global_graph(self):
        """Cache the global synthetic graph for distribution"""
        with torch.no_grad():
            final_adj = self.pge_global.inference(self.syn_x_global)
            self.global_graph_cache = {
                "x": self.syn_x_global.detach().cpu(),
                "adj": final_adj.detach().cpu(),
                "y": self.syn_y_global.detach().cpu()
            }
            
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
                print(f"[Server] Eval: Train {metric_train}, Val {metric_val}, Test {metric_test}")
                
            return eval_output
            
        return override_evaluate

    def send_message(self):
        """Prepare message to send to clients"""
        self.message_pool["server"] = {
            "weights": list(self.task.model.parameters()),
            "global_graph": self.global_graph_cache
        }