import torch
import torch.nn.functional as F
from flcore.fedrgd.client import FedRGDClient
from flcore.fedgm.fedgm_config import config
from torch_sparse import SparseTensor, fill_diag, sum as sparsesum, mul

# Helper: Chuẩn hóa đồ thị nén (Dense)
def robust_normalize_adj(adj, eps=0): 
    adj = torch.clamp(adj, min=0, max=10)
    adj = adj + torch.eye(adj.shape[0], device=adj.device)
    row_sum = torch.sum(adj, 1) + eps
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt = torch.clamp(d_inv_sqrt, min=0, max=10) 
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    adj_norm = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt) 
    return adj_norm

# Helper: Chuẩn hóa đồ thị cục bộ (Sparse)
def normalize_sparse_gcn(adj_t):
    adj_t = fill_diag(adj_t, 1.0) 
    deg = sparsesum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t

class FedGKGClient(FedRGDClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedGKGClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.temperature = config.get('temperature', 1.0) 

    def execute(self):
        server_msg = self.message_pool["server"]
        global_graph = server_msg.get("global_graph")

        local_data = self.task.splitted_data['data']
        l_x, l_y = local_data.x.to(self.device), local_data.y.to(self.device)
        l_edge_index = local_data.edge_index.to(self.device)
        l_mask = self.task.splitted_data['train_mask']
        
        l_edge_weight = torch.ones(l_edge_index.size(1), device=self.device)
        l_adj_sparse = SparseTensor(
            row=l_edge_index[0], col=l_edge_index[1], value=l_edge_weight,
            sparse_sizes=(l_x.size(0), l_x.size(0))
        ).to(self.device)
        l_adj_norm = normalize_sparse_gcn(l_adj_sparse)

        feat_syn, label_syn, adj_syn_norm = None, None, None
        
        if global_graph is not None:
            feat_syn = global_graph['x'].to(self.device).detach()
            label_syn = global_graph['y'].to(self.device) 
            adj_syn = global_graph['adj'].to(self.device).detach()
            
            adj_syn_norm = robust_normalize_adj(adj_syn)

    
        if server_msg.get("weights"):
            with torch.no_grad():
                for (local_param, global_param) in zip(self.task.model.parameters(), server_msg["weights"]):
                    local_param.data.copy_(global_param)
        
        # 4. Training Loop
        self.task.model.train()
        optimizer_cls = self.task.optim 
        
        for epoch in range(self.args.local_epochs):
            optimizer_cls.zero_grad()
            
            # --- A. Local Task Loss ---
            out_local = self.task.model(l_x, l_adj_norm)
            loss_task = F.nll_loss(out_local[l_mask], l_y[l_mask])
            
            loss_total = loss_task

            # --- B. Loss Guidance (Theo công thức bạn cung cấp) ---
            if feat_syn is not None and adj_syn_norm is not None:
                syn_logits = self.task.model(feat_syn, adj_syn_norm)
                
                # 2.KL Divergence (Student)
                student_log_prob = F.log_softmax(syn_logits / self.temperature, dim=1)
                
                if label_syn.dim() == 1: 
                    target_prob = F.one_hot(label_syn, num_classes=syn_logits.size(1)).float()
                else: 
                    target_prob = label_syn
                
                # 4.  KL Divergence
                loss_kd = F.kl_div(student_log_prob, target_prob, reduction='batchmean')                
                loss_guidance = loss_kd * (self.temperature ** 2)
                
                loss_total = loss_total + loss_guidance
            
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.task.model.parameters(), max_norm=1.0)
            optimizer_cls.step()

        # 5. Evaluate & Save Metrics
        local_eval_results = self.task.evaluate(splitted_data=self.task.splitted_data, mute=True)
        self.local_metrics = {
            'loss_train': local_eval_results.get('loss_train', torch.tensor(float('nan'))).item(),
            'accuracy_train': local_eval_results.get('accuracy_train', float('nan')),
            'loss_val': local_eval_results.get('loss_val', torch.tensor(float('nan'))).item(),
            'accuracy_val': local_eval_results.get('accuracy_val', float('nan')),
            'loss_test': local_eval_results.get('loss_test', torch.tensor(float('nan'))).item(),
            'accuracy_test': local_eval_results.get('accuracy_test', float('nan'))
        }
        
        # 6. Local Graph Condensation 
        is_ablation = getattr(self.args, 'ablation', False)
        should_condense = False
        if self.message_pool["round"] > 0:
            should_condense = True
            if is_ablation and self.message_pool["round"] > 1:
                should_condense = False 
        
        if should_condense:
            l_adj_dense = l_adj_sparse.to_dense()
            self._perform_graph_condensation(l_x, l_y, l_adj_dense, l_mask)