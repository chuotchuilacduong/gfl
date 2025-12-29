import torch
from flcore.base import BaseServer

# TODO an algo for graph clustering cls, need to change to node cls 
class FedIGLServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedIGLServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.task.model.prev_grad_1 = None
        self.task.model.prev_grad_2 = None

    def execute(self):
        with torch.no_grad():
            sampled_clients = self.message_pool["sampled_clients"]
            num_tot_samples = sum([self.message_pool[f"client_{cid}"]["num_samples"] for cid in sampled_clients])
            
            # --- 1. Aggregate Weights ---
            for it, client_id in enumerate(sampled_clients):
                client_msg = self.message_pool[f"client_{client_id}"]
                weight = client_msg["num_samples"] / num_tot_samples
                
                for (local_param, global_param) in zip(client_msg["weight"], self.task.model.parameters()):
                    if it == 0:
                        global_param.data.copy_(weight * local_param)
                    else:
                        global_param.data += weight * local_param
            
            # --- 2. Aggregate Gradients (FedIGL Specific) ---
            
            agg_grads_1 = None
            agg_grads_2 = None
            
            for it, client_id in enumerate(sampled_clients):
                client_msg = self.message_pool[f"client_{client_id}"]
                client_weight = client_msg["num_samples"] / num_tot_samples
                
                c_grads_1 = client_msg["grads_1"]
                c_grads_2 = client_msg["grads_2"]
                
                if c_grads_1:
                    if agg_grads_1 is None:
                        agg_grads_1 = [g * client_weight for g in c_grads_1]
                    else:
                        for idx, g in enumerate(c_grads_1):
                            agg_grads_1[idx] += g * client_weight
                            
                if c_grads_2:
                    if agg_grads_2 is None:
                        agg_grads_2 = [g * client_weight for g in c_grads_2]
                    else:
                        for idx, g in enumerate(c_grads_2):
                            agg_grads_2[idx] += g * client_weight

            self.task.model.prev_grad_1 = agg_grads_1
            self.task.model.prev_grad_2 = agg_grads_2

    def send_message(self):
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters()),
            "prev_grad_1": self.task.model.prev_grad_1,
            "prev_grad_2": self.task.model.prev_grad_2
        }