import torch
from flcore.base import BaseClient
from flcore.fedigl.fedigl_gin_model import FedIGL_Node_GIN

class FedIGLClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedIGLClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        
    def execute(self):
        server_msg = self.message_pool["server"]
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), server_msg["weight"]):
                local_param.data.copy_(global_param)
            
            self.task.model.prev_grad_1 = server_msg.get("prev_grad_1")
            self.task.model.prev_grad_2 = server_msg.get("prev_grad_2")

        self.task.model.train()
        
        data = self.task.data.to(self.device)
        train_mask = self.task.train_mask # Tensor boolean [N]
        
        # Optimizer
        optimizer = self.task.optim 
        for epoch in range(self.args.num_epochs):
            optimizer.zero_grad()

            global_pred, x, x1, remaining_edge_index, remaining_edge_weight = self.task.model.forward_global(data)
            
            loss_global = self.task.model.loss(global_pred[train_mask], data.y[train_mask])
            self.task.model.backward(loss_global) 
            pred_local = self.task.model.forward_client(
                x.detach(), 
                x1.detach(), 
                remaining_edge_index.detach(), 
                remaining_edge_weight.detach()
            )
            
            loss_local = self.task.model.loss(pred_local[train_mask], data.y[train_mask])
            loss_local.backward()

            optimizer.step()

    def send_message(self):
 
        grads_1 = None
        grads_2 = None
        
        if hasattr(self.task.model, "stored_grads_1") and self.task.model.stored_grads_1 is not None:
            grads_1 = [g.cpu() for g in self.task.model.stored_grads_1]
            
        if hasattr(self.task.model, "stored_grads_2") and self.task.model.stored_grads_2 is not None:
            grads_2 = [g.cpu() for g in self.task.model.stored_grads_2]

        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()), 
            "grads_1": grads_1,      
            "grads_2": grads_2
        }