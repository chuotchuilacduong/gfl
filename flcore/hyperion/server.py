import torch
import numpy as np
import math
from flcore.base import BaseServer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from flcore.hyperion.utils import calc_cost_matrix_proto, wkd_prototype_loss
import math
# an algo for noise node detection, the performace is low in normal setting? 
class HyperionServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(HyperionServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.num_classes = args.num_classes
        self.num_prototypes_per_class = args.num_prototypes_per_class
        self.prot_dim = args.hid_dim
        
        # Hyperion args
        self.sinkhorn_lambda = args.sinkhorn_lambda
        self.sinkhorn_iter = args.wkd_sinkhorn_iter
        self.temperature = args.wkd_temperature
        self.wkd_logit_loss_weight = args.wkd_logit_weight
        
        
        self.wkd_loss_cosine_decay_epoch = getattr(args, 'wkd_loss_cosine_decay_epoch', 50)
        self.solver_epochs = getattr(args, 'solver_epochs', 100)
        
        self.poisoned_clients = []
        self.benign_clients = []

    def execute(self):
        clients_prototypes = []
        sampled_clients = self.message_pool["sampled_clients"]
        current_round = self.message_pool.get("round", 0)
        for client_id in sampled_clients:
            clients_prototypes.append(self.message_pool[f"client_{client_id}"]["prototypes"])

        # 2. Detect malicious clients
        benign_indices, poisoned_indices = self.detect_malicious_clients(clients_prototypes)      
        benign_client_ids = [sampled_clients[i] for i in benign_indices]
        # 3. Aggregate Model Weights 
        self.aggregate_weights(benign_client_ids)

        # 4. Aggregate & Distill Prototypes
        benign_prototypes = [clients_prototypes[i] for i in benign_indices]
        poisoned_prototypes = [clients_prototypes[i] for i in poisoned_indices]
        initial_global_prototypes = self.aggregate_prototype_gmm(clients_prototypes, benign_indices)
        # 4.2 Distillation
        cost_matrix = calc_cost_matrix_proto(initial_global_prototypes, self.num_prototypes_per_class, self.device)
        
        optimized_prototypes = self.distill_server_by_prototype(
            initial_global_prototypes, 
            benign_prototypes, 
            poisoned_prototypes, 
            cost_matrix,
            epoch=current_round
        )
        # 5. Update Global Model with Prototypes 
        self.update_global_prototypes(optimized_prototypes)

    def aggregate_weights(self, benign_client_ids):
        with torch.no_grad():
            num_tot_samples = sum([self.message_pool[f"client_{cid}"]["num_samples"] for cid in benign_client_ids])
            for it, client_id in enumerate(benign_client_ids):
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                local_params = self.message_pool[f"client_{client_id}"]["weight"]
                
                for (local_param, global_param) in zip(local_params, self.task.model.parameters()):
                    local_param = local_param.to(self.device)
                    if it == 0:
                        global_param.data.copy_(weight * local_param)
                    else:
                        global_param.data += weight * local_param

    def detect_malicious_clients(self, clients_prototypes):
        num_clients = len(clients_prototypes)
        malicious_votes = np.zeros(num_clients)
        
        # Voting qua 9 random seeds
        for vote_idx in range(9): 
            benign_scores = np.zeros(num_clients)
            
            for class_idx in range(self.num_classes):
                try:
                    class_protos = []
                    for client_idx, client_protos in enumerate(clients_prototypes):
                        proto = client_protos[class_idx].detach().cpu().numpy()
                        
                        if proto.ndim > 2:
                            proto = proto.reshape(proto.shape[0], -1)
                        elif proto.ndim == 1:
                            proto = proto.reshape(1, -1)
                            
                        proto_mean = np.mean(proto, axis=0)
                        class_protos.append(proto_mean)
                    
                    class_protos = np.array(class_protos)
                    
                    scaler = StandardScaler()
                    class_protos_normalized = scaler.fit_transform(class_protos)
                    
                    if len(class_protos) <= 2 or np.all(np.std(class_protos_normalized, axis=0) < 1e-6):
                        continue
                    
                    gmm = GaussianMixture(
                        n_components=min(2, len(class_protos)-1),  
                        random_state=vote_idx,  
                        covariance_type='full',
                        max_iter=100,
                        n_init=3 
                    )
                    
                    cluster_labels = gmm.fit_predict(class_protos_normalized)
                    proba = gmm.predict_proba(class_protos_normalized)

                    counts = np.bincount(cluster_labels)
                    benign_cluster = np.argmax(counts)
                    
                    benign_scores += proba[:, benign_cluster]
                    
                except Exception as e:
                    continue
            
            if np.all(benign_scores == 0):
                continue
                
            threshold = np.mean(benign_scores) - np.std(benign_scores)
            max_poisoned = int(num_clients * 0.4)
            sorted_indices = np.argsort(benign_scores)
            
            current_poisoned = sorted_indices[:min(max_poisoned, len(np.where(benign_scores < threshold)[0]))]
            malicious_votes[current_poisoned] += 1
        
        vote_threshold = 9 / 2  
        final_poisoned = np.where(malicious_votes > vote_threshold)[0].tolist()
        
        if not final_poisoned:

            max_select = int(num_clients * 0.3)
            sorted_by_votes = np.argsort(-malicious_votes) 
            final_poisoned = sorted_by_votes[:max_select].tolist()
            
            if np.all(malicious_votes[final_poisoned] == 0):
                final_poisoned = []
        
        self.poisoned_clients = final_poisoned
        self.benign_clients = list(set(range(num_clients)) - set(final_poisoned))
        
        return self.benign_clients, self.poisoned_clients

    def aggregate_prototype_gmm(self, clients_prototypes, benign_indices):
        aggregated_prototypes = {}
        for class_idx in range(self.num_classes):
            class_protos_list = []
            for idx in benign_indices:
                class_protos_list.append(clients_prototypes[idx][class_idx])
            
            class_protos = torch.cat(class_protos_list, dim=0)
            gmm = GaussianMixture(n_components=self.num_prototypes_per_class, random_state=42, covariance_type='full')
            gmm.fit(class_protos.numpy())
            aggregated_prototypes[class_idx] = torch.tensor(gmm.means_).clone().detach().requires_grad_(True)
        return aggregated_prototypes
    def convert_prototypes_to_tensor(self, prototypes_list):
        converted_prototypes = []
        for client_protos in prototypes_list:
            proto_tensors = []
            for class_idx in range(self.num_classes):
                proto_tensors.append(client_protos[class_idx])
            
            client_proto_tensor = torch.cat(proto_tensors, dim=0)  
            
            client_proto_tensor = client_proto_tensor.reshape(
                self.num_classes * self.num_prototypes_per_class, 
                client_proto_tensor.shape[-1]
            ).to(torch.float32).to(self.device) 
            converted_prototypes.append(client_proto_tensor)
        
        return converted_prototypes
    def distill_server_by_prototype(self, initial_global_prototypes, benign_prototypes, poisoned_prototypes, cost_matrix, lr=0.01, max_steps=3, **kwargs):
        self.temperature = self.args.wkd_temperature   
        self.sinkhorn_lambda = self.args.sinkhorn_lambda
        self.sinkhorn_iter = self.args.wkd_sinkhorn_iter 
        self.loss_cosine_decay_epoch = getattr(self.args, 'wkd_loss_cosine_decay_epoch', 50) 
        self.wkd_logit_loss_weight = self.args.wkd_logit_weight 
        self.solver_epochs = getattr(self.args, 'solver_epochs', 100)

        decay_start_epoch = self.loss_cosine_decay_epoch
        epoch = kwargs.get('epoch', 0)
        
        if epoch > decay_start_epoch:
            self.wkd_logit_loss_weight_1 = 0.5 * self.wkd_logit_loss_weight * (1 + math.cos((epoch - decay_start_epoch)/(self.solver_epochs - decay_start_epoch) * math.pi))
        else:
            self.wkd_logit_loss_weight_1 = self.wkd_logit_loss_weight

        trainable_prototypes = {}
        for class_idx in range(self.num_classes):
            trainable_prototypes[class_idx] = initial_global_prototypes[class_idx].clone().detach().to(self.device).requires_grad_(True)
        
        logits_teacher = torch.stack([trainable_prototypes[class_idx] for class_idx in range(self.num_classes)]) \
                              .reshape(self.num_classes * self.num_prototypes_per_class, -1) \
                              .to(torch.float32)

        benign_prototypes_reshaped = self.convert_prototypes_to_tensor(benign_prototypes)
        poisoned_prototypes_reshaped = self.convert_prototypes_to_tensor(poisoned_prototypes)

        optimizer = torch.optim.Adam([proto for proto in trainable_prototypes.values()], lr=lr)

        for step in range(max_steps):  
            total_loss = 0.0
            
            for class_idx in range(self.num_classes * self.num_prototypes_per_class):
                optimizer.zero_grad()
                
                for logits_good_student in benign_prototypes_reshaped:
                    loss_wkd_logit = wkd_prototype_loss(
                        logits_good_student[class_idx], 
                        logits_teacher[class_idx], 
                        self.temperature,
                        self.wkd_logit_loss_weight_1, 
                        cost_matrix, 
                        self.sinkhorn_lambda, 
                        self.sinkhorn_iter
                    )
                    total_loss += loss_wkd_logit
                
                for logits_bad_student in poisoned_prototypes_reshaped:
                    loss_wkd_logit = wkd_prototype_loss(
                        logits_bad_student[class_idx], 
                        logits_teacher[class_idx], 
                        self.temperature,
                        self.wkd_logit_loss_weight_1, 
                        cost_matrix, 
                        self.sinkhorn_lambda, 
                        self.sinkhorn_iter
                    )
                    total_loss -= loss_wkd_logit  
                
                total_loss.backward(retain_graph=True)
                
                torch.nn.utils.clip_grad_norm_([proto for proto in trainable_prototypes.values()], max_norm=1.0)
                optimizer.step()
                
                total_loss = 0.0 
        optimized_prototypes = {}
        for class_idx in range(self.num_classes):
            proto = trainable_prototypes[class_idx].detach()
            proto_norm = torch.norm(proto, p=2, dim=1, keepdim=True)
            optimized_prototypes[class_idx] = (proto / (proto_norm + 1e-8)).float()
        
        return optimized_prototypes

    def update_global_prototypes(self, global_prototypes):
        prototype_vectors = []
        for class_idx in range(self.num_classes):
            if class_idx in global_prototypes:
                prototype_vectors.append(global_prototypes[class_idx])
        if prototype_vectors:
            prototype_vectors = torch.cat(prototype_vectors, dim=0).to(torch.float32)
            self.task.model.prototype_layer.prototype_vectors.data = prototype_vectors.to(self.device)
    def send_message(self):
        weights = [p.data.clone().detach().cpu() for p in self.task.model.parameters()]
        self.message_pool["server"] = {
            "weight": weights,
            "poisoned_clients": self.poisoned_clients
        }