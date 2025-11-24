import torch
import random
from data.distributed_dataset_loader import FGLDataset
from utils.basic_utils import load_client, load_server
from utils.logger import Logger
import wandb
import numpy as np
class FGLTrainer:
    """
    Federated Graph Learning Trainer class to manage the training and evaluation process.

    Attributes:
        args (Namespace): Arguments containing model and training configurations.
        message_pool (dict): Dictionary to manage messages between clients and server.
        device (torch.device): Device to run the computations on.
        clients (list): List of client instances.
        server (object): Server instance.
        evaluation_result (dict): Dictionary to store the best evaluation results.
        logger (Logger): Logger instance to log training and evaluation metrics.
    """
    
    
    def __init__(self, args):
        """
        Initialize the FGLTrainer with provided arguments and dataset.

        Args:
            args (Namespace): Arguments containing model and training configurations.
        """
        self.args = args
        self.message_pool = {}
        fgl_dataset = FGLDataset(args)
        self.device = torch.device(f"cuda:{args.gpuid}" if (torch.cuda.is_available() and args.use_cuda) else "cpu")
        self.clients = [load_client(args, client_id, fgl_dataset.local_data[client_id], fgl_dataset.processed_dir, self.message_pool, self.device) for client_id in range(self.args.num_clients)]
        self.server = load_server(args, fgl_dataset.global_data, fgl_dataset.processed_dir, self.message_pool, self.device)
        sorted_datasets = sorted(args.dataset)
        group_name = f"Dataset={'_'.join(sorted_datasets)}-Sim={args.simulation_mode}-Clients={args.num_clients}"
        run_name = f"{args.fl_algorithm}"

        # wandb.init(
        #         project="ACGraphFL",
                
        #         config=vars(args),
        #         group=group_name,  
        #         name=run_name,     
        #     )
        self.evaluation_result = {"best_round":0}
        if self.args.task in ["graph_cls", "graph_reg", "node_cls", "link_pred"]:
            for metric in self.args.metrics:
                self.evaluation_result[f"best_val_{metric}"] = 0
                self.evaluation_result[f"best_test_{metric}"] = 0
        elif self.args.task in ["node_clust"]:
            for metric in self.args.metrics:
                self.evaluation_result[f"best_{metric}"] = 0
        

        self.logger = Logger(args, self.message_pool, fgl_dataset.processed_dir, self.server.personalized)
        
  
    def train(self):
        """
        Train the model over a specified number of rounds, performing federated learning with the clients.
        """
        for round_id in range(self.args.num_rounds):
            sampled_clients = sorted(random.sample(list(range(self.args.num_clients)), int(self.args.num_clients * self.args.client_frac)))
            print(f"round # {round_id}\t\tsampled_clients: {sampled_clients}")
            self.message_pool["round"] = round_id
            self.message_pool["sampled_clients"] = sampled_clients
            self.server.send_message()
            client_metrics_list = []
            for client_id in sampled_clients:
                self.clients[client_id].execute()
                self.clients[client_id].send_message()
                
                client_msg = self.message_pool.get(f"client_{client_id}", {})
                if "local_metrics" in client_msg:
                    client_metrics_list.append(client_msg["local_metrics"])

            avg_client_metrics = {}
            if client_metrics_list:
                keys = client_metrics_list[0].keys()
                for key in keys:
                    valid_metrics = [m[key] for m in client_metrics_list if key in m and not np.isnan(m[key])]
                    if valid_metrics:
                        avg_client_metrics[key] = np.mean(valid_metrics)
                    else:
                        avg_client_metrics[key] = float('nan')
            self.server.execute()
            
            server_matching_loss = float('nan')
            if hasattr(self.server, 'last_gradient_match_loss'):
                server_matching_loss = self.server.last_gradient_match_loss
            
            print(f"  LOCAL Client Metrics (Avg): "
                  f"Train Loss: {avg_client_metrics.get('loss_train', float('nan')):.4f}, Train Acc: {avg_client_metrics.get('accuracy_train', float('nan')):.4f} | "
                  f"Val Loss: {avg_client_metrics.get('loss_val', float('nan')):.4f}, Val Acc: {avg_client_metrics.get('accuracy_val', float('nan')):.4f} | "
                  f"Test Loss: {avg_client_metrics.get('loss_test', float('nan')):.4f}, Test Acc: {avg_client_metrics.get('accuracy_test', float('nan')):.4f}")
            print(f"  SERVER Metrics: "
                  f"Matching Loss: {server_matching_loss:.4f}")

            log_data = {"current_round": round_id}
            log_data.update({f"local_{key}": val for key, val in avg_client_metrics.items()})
            log_data['matching_loss'] = server_matching_loss
           
            avg_train_loss_for_log = avg_client_metrics.get('loss_train', float('nan'))
            
           
            global_eval_results = self.evaluate(avg_train_loss=avg_train_loss_for_log)
            
            log_data.update(global_eval_results) 
            self.logger.add_log(log_data)
            # wandb.log(log_data)
            print("-"*50)
            
        self.logger.save()
        final_summary = {}
        for metric in self.args.metrics:
             if f"best_val_{metric}" in self.evaluation_result:
                 final_summary[f"final_best_val/{metric}"] = self.evaluation_result[f"best_val_{metric}"]
                 final_summary[f"final_best_test/{metric}"] = self.evaluation_result[f"best_test_{metric}"]
             elif f"best_{metric}" in self.evaluation_result:
                 final_summary[f"final_best/{metric}"] = self.evaluation_result[f"best_{metric}"]
        final_summary["final_best_round"] = self.evaluation_result['best_round']
        # wandb.summary.update(final_summary)
        # wandb.finish()
        
        
        
    def evaluate(self, avg_train_loss=float('nan')):
        """
        Evaluate the model based on the specified evaluation mode and task.

        Raises:
            ValueError: If the evaluation mode is not supported by the personalized algorithm.
        """
        evaluation_result = {"current_round": self.message_pool["round"]}
        # avg_train_loss (truyền vào) sẽ được dùng ở cuối
        
        avg_loss_val = 0.0
        avg_loss_test = 0.0
        current_round = evaluation_result['current_round']
        
    
        if self.args.task in ["graph_cls", "graph_reg", "node_cls", "link_pred"]:
            
            for metric in self.args.metrics:
                evaluation_result[f"current_val_{metric}"] = 0
                evaluation_result[f"current_test_{metric}"] = 0
                
        elif self.args.task in ["node_clust"]:
            avg_loss_total = 0.0
            for metric in self.args.metrics:
                evaluation_result[f"current_{metric}"] = 0
                
        
        tot_samples = 0
        one_time_infer = False
        
        
        for client_id in range(self.args.num_clients):
            if self.args.evaluation_mode == "local_model_on_local_data":
                num_samples = self.clients[client_id].task.num_samples
                result = self.clients[client_id].task.evaluate(mute=True)
            elif self.args.evaluation_mode == "local_model_on_global_data":
                num_samples = self.server.task.num_samples
                result = self.clients[client_id].task.evaluate(self.server.task.splitted_data,mute=True)
            elif self.args.evaluation_mode == "global_model_on_local_data":
                num_samples = self.clients[client_id].task.num_samples
                if self.server.personalized:
                    raise ValueError(f"personalized algorithm {self.args.fl_algorithm} doesn't support global model evaluation.")
                result = self.server.task.evaluate(self.clients[client_id].task.splitted_data,mute=True)
            elif self.args.evaluation_mode == "global_model_on_global_data":
                num_samples = self.server.task.num_samples
                if self.server.personalized:
                    raise ValueError(f"personalized algorithm {self.args.fl_algorithm} doesn't support global model evaluation.")
                # only one-time infer
                one_time_infer = True
                result = self.server.task.evaluate(mute=True)
            
            if self.args.task in ["graph_cls", "graph_reg", "node_cls", "link_pred"]:
                avg_loss_val += result.get('loss_val', torch.tensor(0.0)).item() * num_samples
                avg_loss_test += result.get('loss_test', torch.tensor(0.0)).item() * num_samples
                for metric in self.args.metrics:
                    
                    val_metric, test_metric = result[f"{metric}_val"], result[f"{metric}_test"]
                    evaluation_result[f"current_val_{metric}"] += val_metric * num_samples
                    evaluation_result[f"current_test_{metric}"] += test_metric * num_samples
            elif self.args.task in ["node_clust"]:
                avg_loss_total += result.get('loss', torch.tensor(0.0)).item() * num_samples
                for metric in self.args.metrics:
                    metric_value = result[f"{metric}"]
                    evaluation_result[f"current_{metric}"] += metric_value * num_samples
                
            if one_time_infer:
                tot_samples = num_samples
                break
            else:
                tot_samples += num_samples
        
        
        
        if self.args.task in ["graph_cls", "graph_reg", "node_cls", "link_pred"]:
            avg_loss_val /= tot_samples
            avg_loss_test /= tot_samples
            for metric in self.args.metrics:
                evaluation_result[f"current_val_{metric}"] /= tot_samples
                evaluation_result[f"current_test_{metric}"] /= tot_samples
                
            if evaluation_result[f"current_val_{self.args.metrics[0]}"] > self.evaluation_result[f"best_val_{self.args.metrics[0]}"]:
                for metric in self.args.metrics:
                    self.evaluation_result[f"best_val_{metric}"] = evaluation_result[f"current_val_{metric}"]
                    self.evaluation_result[f"best_test_{metric}"] = evaluation_result[f"current_test_{metric}"]
                self.evaluation_result[f"best_round"] = evaluation_result[f"current_round"]
            
            current_output = f"  GLOBAL Eval (Round {evaluation_result['current_round']}): " + \
                    "\t".join([f"Val_{metric}: {evaluation_result.get(f'current_val_{metric}', float('nan')):.4f}\tTest_{metric}: {evaluation_result.get(f'current_test_{metric}', float('nan')):.4f}" for metric in self.args.metrics]) + \
                    f"\tAvg_Val_Loss: {avg_loss_val:.4f}\tAvg_Test_Loss: {avg_loss_test:.4f}" # Bỏ avg_train_loss
            best_output = f"  BEST (Round {self.evaluation_result['best_round']}):   " + \
                "\t".join([f"Best_Val_{metric}: {self.evaluation_result[f'best_val_{metric}']:.4f}\tBest_Test_{metric}: {self.evaluation_result[f'best_test_{metric}']:.4f}" for metric in self.args.metrics])
    
            print(current_output)
            print(best_output)
        
        else:
            avg_loss_total /= tot_samples
            for metric in self.args.metrics:
                evaluation_result[f"current_{metric}"] /= tot_samples
        
            if evaluation_result[f"current_{self.args.metrics[0]}"] > self.evaluation_result[f"best_{self.args.metrics[0]}"]:
                for metric in self.args.metrics:
                    self.evaluation_result[f"best_{metric}"] = evaluation_result[f"current_{metric}"]
                self.evaluation_result[f"best_round"] = evaluation_result[f"current_round"]
            
            current_output = f"  GLOBAL Eval (Round {evaluation_result['current_round']}): " + \
                    "\t".join([f"curr_{metric}: {evaluation_result.get(f'current_{metric}', float('nan')):.4f}" for metric in self.args.metrics]) + \
                    f"\tAvg_Total_Loss: {avg_loss_total:.4f}" # Bỏ avg_train_loss
            best_output = f"best_round: {self.evaluation_result['best_round']}\t" + \
                "\t".join([f"best_{metric}: {self.evaluation_result[f'best_{metric}']:.4f}" for metric in self.args.metrics])
        
            print(current_output)
            print(best_output)
            
        evaluation_result['loss_train'] = avg_train_loss 
        if self.args.task in ["graph_cls", "graph_reg", "node_cls", "link_pred"]:
            evaluation_result['loss_val'] = avg_loss_val 
            evaluation_result['loss_test'] = avg_loss_test 
        elif self.args.task in ["node_clust"]:
            evaluation_result['loss_total'] = avg_loss_total 
        
        
        return evaluation_result
            
    
        
        
        
     