import torch
from flcore.base import BaseClient
import torch.nn.functional as F
from flcore.fedigl.fedigl_gin_model import FedIGL_GIN
class FedIGLClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedIGLClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3
        self.accum_grads_1 = {}
        self.accum_grads_2 = {}
        self.num_batches_tracked = 0
    def execute(self):
        server_msg = self.message_pool["server"]
        
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), server_msg["weight"]):
                local_param.data.copy_(global_param)
        
        self.task.model.prev_grad_1 = server_msg.get("prev_grad_1", None)
        self.task.model.prev_grad_2 = server_msg.get("prev_grad_2", None)

        self.task.model.train()
        is_node_level = False
        train_mask = None
        
        if hasattr(self.task, 'train_loader'):
            train_loader = self.task.train_loader
        else:
            is_node_level = True
            if hasattr(self.task, 'processed_data') and self.task.processed_data is not None:
                data = self.task.processed_data['data']
                train_mask = self.task.processed_data['train_mask'].to(self.device)
            else:
                data = self.task.data
                train_mask = self.task.train_mask.to(self.device)
            
            if not hasattr(data, 'batch') or data.batch is None:
                data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
            
            train_loader = [data] 
        if hasattr(self, 'optimizer'):
            optimizer = self.optimizer
        elif hasattr(self.task, 'optim'):
            optimizer = self.task.optim
        else:
            optimizer = torch.optim.Adam(self.task.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        self.accum_grads_1 = {}
        self.accum_grads_2 = {}
        self.num_batches_tracked = 0

        for epoch in range(self.args.num_epochs): 
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                global_pred, x, x1, remaining_edge_index, remaining_edge_weight, batch2 = self.task.model.forward_global(batch)
                label = batch.y
                if is_node_level and train_mask is not None:
                    
                    loss = F.nll_loss(global_pred[train_mask], label[train_mask])
                else:
                    loss = self.task.model.loss(global_pred, label)
                self.task.model.backward(loss)
                
                pred = self.task.model.forward_client(x.detach(), x1.detach(), remaining_edge_index.detach(),
                                            remaining_edge_weight.detach(), batch2)
                loss2 = self.task.model.loss(pred, label)
                
                loss2.backward()
                
                optimizer.step()

    def send_message(self):
        grads_1 = []
        if hasattr(self.task.model, "graph_convs"):
            grads_1 = [param.grad.clone() for param in self.task.model.graph_convs.parameters() if param.grad is not None]

        grads_2 = []
        if hasattr(self.task.model, "graph_convs4"):
            grads_2 = [param.grad.clone() for param in self.task.model.graph_convs4.parameters() if param.grad is not None]

        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
            "grads_1": grads_1,
            "grads_2": grads_2
        }