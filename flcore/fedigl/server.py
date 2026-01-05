import torch
from flcore.base import BaseServer

class FedIGLServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedIGLServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.prev_grad_1 = None
        self.prev_grad_2 = None

    def execute(self):
        with torch.no_grad():
            sampled_clients = self.message_pool["sampled_clients"]
            num_tot_samples = sum([self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in sampled_clients])
            
            # Reset temporary aggregators
            agg_grads_1 = None
            agg_grads_2 = None

            for it, client_id in enumerate(sampled_clients):
                client_data = self.message_pool[f"client_{client_id}"]
                weight = client_data["num_samples"] / num_tot_samples
                
                # 1. Aggregate Model Weights 
                for (local_param, global_param) in zip(client_data["weight"], self.task.model.parameters()):
                    if it == 0:
                        global_param.data.copy_(weight * local_param)
                    else:
                        global_param.data += weight * local_param

                local_grads_1 = client_data.get("grads_1")
                local_grads_2 = client_data.get("grads_2")

                if local_grads_1 is not None:
                    if agg_grads_1 is None:
                        agg_grads_1 = [weight * g for g in local_grads_1]
                    else:
                        for idx, g in enumerate(local_grads_1):
                            agg_grads_1[idx] += weight * g
                
                if local_grads_2 is not None:
                    if agg_grads_2 is None:
                        agg_grads_2 = [weight * g for g in local_grads_2]
                    else:
                        for idx, g in enumerate(local_grads_2):
                            agg_grads_2[idx] += weight * g

            self.prev_grad_1 = agg_grads_1
            self.prev_grad_2 = agg_grads_2
            
            self.task.model.prev_grad_1 = self.prev_grad_1
            self.task.model.prev_grad_2 = self.prev_grad_2

    def send_message(self):
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters()),
            "prev_grad_1": self.prev_grad_1,
            "prev_grad_2": self.prev_grad_2
        }