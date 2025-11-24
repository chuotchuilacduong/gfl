import torch
import torch.nn as nn
from flcore.base import BaseClient

class FedAvgClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedAvgClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.local_metrics = {}
        
    def execute(self):
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                local_param.data.copy_(global_param)

        self.task.train()
        local_eval_results = self.task.evaluate(splitted_data=self.task.splitted_data, mute=True)
        
        # Lưu trữ các số liệu (giống hệt như cách fedgm_hybrid thực hiện)
        self.local_metrics = {
            'loss_train': local_eval_results.get('loss_train', torch.tensor(float('nan'))).item(),
            'accuracy_train': local_eval_results.get('accuracy_train', float('nan')),
            'loss_val': local_eval_results.get('loss_val', torch.tensor(float('nan'))).item(),
            'accuracy_val': local_eval_results.get('accuracy_val', float('nan')),
            'loss_test': local_eval_results.get('loss_test', torch.tensor(float('nan'))).item(),
            'accuracy_test': local_eval_results.get('accuracy_test', float('nan'))
        }
        
    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters()),
                "local_metrics": self.local_metrics
            }
        
        
        