import torch
import torch.nn.functional as F
import copy
from flcore.base import BaseClient
from flcore.fedaux.models import GNNAUX

class FedAuxClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedAuxClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        
        if hasattr(self.task.data, 'num_node_features'):
            n_feat = self.task.data.num_node_features
        else:
            n_feat = self.task.data.x.shape[1]
            
        n_clss = args.num_classes
        n_dims = args.hid_dim
        
        self.task.model = GNNAUX(n_feat, n_dims, n_clss, args).to(device)
        
        self.optimizer = torch.optim.Adam(
            self.task.model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        
        self.prev_w = None

    def execute(self):
        server_msg = self.message_pool.get(f"server_{self.client_id}")
        if server_msg is None:
            server_msg = self.message_pool.get("server")
            
        if server_msg is None:
             pass
        else:
            self.task.model.load_state_dict(server_msg["weight"])
            
            self.prev_w = {
                name: param.clone().detach() 
                for name, param in self.task.model.named_parameters()
            }

        self.task.model.train()
        data = self.task.data.to(self.device)
        curr_rnd = self.message_pool.get("round", 0)

        for epoch in range(self.args.num_epochs):
            self.optimizer.zero_grad()
            _, output = self.task.model(data)
            
            loss = F.cross_entropy(output[self.task.train_mask], data.y[self.task.train_mask])
            
            if curr_rnd > 0 and self.prev_w is not None:
                reg_loss = 0.0
                for name, param in self.task.model.named_parameters():
                    if ('aux' in name or 'conv' in name or 'clsif' in name) and name in self.prev_w:
                        reg_loss += torch.norm(param - self.prev_w[name], 2)
                
                loss += reg_loss * self.args.loc_l2
                
            loss.backward()
            self.optimizer.step()

    def send_message(self):
    
        aux_val = None
        for name, param in self.task.model.named_parameters():
            if 'aux' in name:
                aux_val = param.data.clone() 
                break
        
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": self.task.model.state_dict(),
            "aux": aux_val 
        }