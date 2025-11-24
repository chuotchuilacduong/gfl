import torch
from flcore.fedgm.client import FedGMClient
from model.gcn import GCN_kipf 
from utils.task_utils import load_node_edge_level_default_model

class FedGMHybridClient(FedGMClient):
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedGMHybridClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.task_model_b = load_node_edge_level_default_model(
            args, 
            input_dim=self.task.num_feats, 
            output_dim=self.task.num_global_classes,
            client_id=client_id
        ).to(device)
        self.task.load_custom_model(self.task_model_b)
        self.local_metrics = {}
        self.task.override_evaluate = None
    def execute(self):
        if self.message_pool["round"] == 0:           
            return
        server_payload = self.message_pool["server"]
        weights_a = server_payload["fedgm_weights"]  
        weights_b = server_payload["fedavg_weights"] 
        G_syn_global = server_payload["syn_graph_global"]
        #model A - FedGM
        with torch.no_grad():
            for (local_param, global_param) in zip(self.model.parameters(), weights_a):
                local_param.data.copy_(global_param)
        self.get_gradient(self.task.splitted_data)
        #model B - FedAvg
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), weights_b):
                local_param.data.copy_(global_param)
        self.task.train(
            splitted_data=self.task.splitted_data, 
            G_syn_global=G_syn_global
        )
        local_eval_results = self.task.evaluate(splitted_data=self.task.splitted_data, mute=True)
        
        self.local_metrics = {
            'loss_train': local_eval_results.get('loss_train', torch.tensor(float('nan'))).item(),
            'accuracy_train': local_eval_results.get('accuracy_train', float('nan')),
            'loss_val': local_eval_results.get('loss_val', torch.tensor(float('nan'))).item(),
            'accuracy_val': local_eval_results.get('accuracy_val', float('nan')),
            'loss_test': local_eval_results.get('loss_test', torch.tensor(float('nan'))).item(),
            'accuracy_test': local_eval_results.get('accuracy_test', float('nan'))
        }
    def send_message(self):
        
        if self.message_pool["round"] == 0:
        
            super().send_message()
        else:
            self.message_pool[f"client_{self.client_id}"] = {
                # 1.  FedGM
                "fedgm_gradient": self.local_class_gradient,
                "local_num_class_dict": self.num_class_dict,
                
                # 2.  FedAvg
                "num_samples": self.task.num_samples, #
                "fedavg_weights": list(self.task.model.parameters()),
                "local_metrics": self.local_metrics
            }