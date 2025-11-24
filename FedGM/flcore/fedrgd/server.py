import torch
from flcore.fedgm.server import FedGMServer
from flcore.fedgm.utils import match_loss, normalize_adj_tensor
from flcore.fedgm.pge import PGE
from flcore.fedgm.fedgm_config import config
from model.gcn import GCN_kipf

class FedRGDServer(FedGMServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedRGDServer, self).__init__(args, global_data, data_dir, message_pool, device)
        
        self.num_global_syn_nodes = args.num_global_syn_nodes
        self.pge_global = PGE(nfeat=self.task.num_feats, nnodes=self.num_global_syn_nodes, device=device, args=args).to(device)
        self.syn_x_global = torch.randn(self.num_global_syn_nodes, self.task.num_feats, requires_grad=True, device=device)
        self.syn_y_global = torch.LongTensor([i % self.task.num_global_classes for i in range(self.num_global_syn_nodes)]).to(device)
        
        self.optimizer_global = torch.optim.Adam([
            {'params': [self.syn_x_global], 'lr': args.lr_feat},
            {'params': self.pge_global.parameters(), 'lr': args.lr_adj}
        ])
        
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
        self.global_graph_cache = None

    
    
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
        self.aggregate_classification_weights()
        self.last_gradient_match_loss = float('nan')
        
        if self.message_pool["round"] > 0:
            self.model_cond.train()
            for p in self.model_cond.parameters(): p.requires_grad = True           
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

            if not input_graphs: return

            for it in range(self.args.server_condense_iters):
                self.model_cond.initialize() 
                
                self.optimizer_global.zero_grad()
                adj_syn_global = self.pge_global(self.syn_x_global)
                adj_syn_norm = normalize_adj_tensor(adj_syn_global, sparse=False)
                out_syn_global = self.model_cond(self.syn_x_global, adj_syn_norm)
                
                total_iter_loss = 0.0
                
                for client_graph in input_graphs:
                    self.model_cond.zero_grad() 
                    
                    c_x = client_graph["x"].to(self.device)
                    c_adj = client_graph["adj"].to(self.device) 
                    c_y = client_graph["y"].to(self.device)
                    c_adj_norm = normalize_adj_tensor(c_adj, sparse=False)
                    
                    out_target = self.model_cond(c_x, c_adj_norm)
                    
                    unique_classes = torch.unique(c_y)
                    client_loss = 0.0 
                    
                    for c in unique_classes:
                        c = c.item()
                        if c not in global_syn_class_indices: continue
                        mask_c = (c_y == c)
                        loss_target_c = torch.nn.functional.nll_loss(out_target[mask_c], c_y[mask_c])
                        gw_target_c = torch.autograd.grad(loss_target_c, self.model_cond.parameters(), retain_graph=True)
                        gw_target_c = [g.detach() for g in gw_target_c] # Detach ngay để tiết kiệm bộ nhớ

                        # 2. Syn Gradient
                        idx_c = global_syn_class_indices[c]
                        loss_syn_c = torch.nn.functional.nll_loss(out_syn_global[idx_c], self.syn_y_global[idx_c])
                        gw_syn_c = torch.autograd.grad(loss_syn_c, self.model_cond.parameters(), create_graph=True, retain_graph=True)

                        # 3. Match Loss
                        client_loss += match_loss(gw_syn_c, gw_target_c, dis_metric=config["dis_metric"], device=self.device)
                    
                    total_iter_loss += client_loss

                if len(input_graphs) > 0:
                    total_iter_loss = total_iter_loss / len(input_graphs)

                total_iter_loss.backward()
                self.optimizer_global.step()

            for p in self.model_cond.parameters(): p.requires_grad = True
            
            with torch.no_grad():
                final_adj = self.pge_global.inference(self.syn_x_global)
                self.global_graph_cache = {
                    "x": self.syn_x_global.detach().cpu(),
                    "adj": final_adj.detach().cpu(),
                    "y": self.syn_y_global.detach().cpu()
                }

    def send_message(self):
        self.message_pool["server"] = {
            "weights": list(self.task.model.parameters()),
            "global_graph": self.global_graph_cache
        }