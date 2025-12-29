import torch
import torch.nn.functional as F
from flcore.base import BaseClient
from torch_geometric.utils import dropout_adj, mask_feature
import copy
import numpy as np
from flcore.hyperion.utils import prune_training_set

class HyperionClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(HyperionClient, self).__init__(args, client_id, data, data_dir, message_pool, device)       
        
        self.num_prototypes_per_class = args.num_prototypes_per_class
        self.num_classes = args.num_classes
        self.clst = args.clst
        self.sep = args.sep 
        self.proto_contrast_weight = args.proto_contrast_weight
        self.alpha = args.alpha
        self.tau = args.tau
        self.p = args.p  
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.task.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def execute(self):
        self.task.model.train()
        self.task.model.to(self.device)
        data = self.task.data.to(self.device)
        if not hasattr(data, 'train_mask') or data.train_mask is None:
            data.train_mask = self.task.train_mask.to(self.device)
            
        server_msg = self.message_pool.get("server", {})
        if "weight" in server_msg:
            server_weights = server_msg["weight"]
            with torch.no_grad():
                for client_param, server_param in zip(self.task.model.parameters(), server_weights):
                    client_param.data.copy_(server_param.data)
        poisoned_list = server_msg.get("poisoned_clients", [])
        current_round = self.message_pool.get("round", 0)
        pruning_epochs = getattr(self.args, 'pruning_epochs', 10)
        
        if (self.client_id in poisoned_list) and \
           (current_round > pruning_epochs) and \
           (current_round % 5 == 0):
            
            with torch.no_grad():
                model_out = self.task.model(data)
                if isinstance(model_out, tuple):
                     distances = model_out[-1] 
                else:
                     distances = model_out
            
            retain_ratio = getattr(self.args, 'retain_ratio', 0.8)
            data = prune_training_set(data, self.task.model, distances, retain_ratio, self.device)
            
            self.task.data = data
            self.task.train_mask = data.train_mask
            
        for epoch in range(self.args.num_epochs):
            self.optimizer.zero_grad()
            
            # Data Augmentation (Hyperion logic)
            data1 = copy.deepcopy(data)
            data1.edge_index, _ = dropout_adj(data.edge_index, p=self.p)
            data1.x = mask_feature(data.x, p=self.p)[0]
            
            pred, virtual_label, prot_nce_loss, graph_emb, distances = self.task.model(data)
            pred1, _, _, _, _ = self.task.model(data1)

            proto_ident = self.task.model.prototype_layer.prototype_class_identity
            
            train_labels = data.y[data.train_mask]
            
            prototypes_of_correct_class = torch.t(proto_ident[:, train_labels].bool())
            train_distances = distances[data.train_mask]
            
            # Clustering cost
            cluster_cost = torch.mean(torch.min(train_distances[prototypes_of_correct_class].reshape(-1, self.num_prototypes_per_class), dim=1)[0])     
            
            # Separation cost
            eps = 1e-6
            separation_cost = torch.mean(1.0 / (torch.min(train_distances[~prototypes_of_correct_class].reshape(-1, (self.num_classes - 1) * self.num_prototypes_per_class), dim=1)[0] + eps))   
            
            # Prototype Contrastive Loss
            prototype_vectors = self.task.model.get_prototype_vectors()
            proto_contrast_loss = self.prototype_contrastive_loss(prototype_vectors)
            
            # Dynamic CE Loss
            dynamic_ce_loss = self.dynamic_cross_entropy_loss(
                pred1[data.train_mask], 
                pred[data.train_mask], 
                data.y[data.train_mask]
            )

            total_loss = self.clst * cluster_cost + \
                         self.sep * separation_cost + \
                         self.proto_contrast_weight * proto_contrast_loss + \
                         self.alpha * dynamic_ce_loss

            total_loss.backward()
            self.optimizer.step()

    def send_message(self):
        prototypes_by_class = self.get_prototypes()
        weights = [p.data.clone().detach().cpu() for p in self.task.model.parameters()]
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.train_mask.sum().item(),
            "weight": weights,
            "prototypes": prototypes_by_class 
        }

    def prototype_contrastive_loss(self, prototype_vectors):
        num_prototypes = prototype_vectors.shape[0]
        prototype_vectors_norm = F.normalize(prototype_vectors, p=2, dim=1)
        sim_matrix = torch.mm(prototype_vectors_norm, prototype_vectors_norm.t()) / self.tau
        
        # Masking logic... 
        same_class_mask = torch.zeros((num_prototypes, num_prototypes), device=prototype_vectors.device)
        for class_idx in range(self.num_classes):
            start_idx = class_idx * self.num_prototypes_per_class
            end_idx = (class_idx + 1) * self.num_prototypes_per_class
            same_class_mask[start_idx:end_idx, start_idx:end_idx] = 1
        same_class_mask = same_class_mask - torch.eye(num_prototypes, device=prototype_vectors.device)
        diff_class_mask = 1 - same_class_mask - torch.eye(num_prototypes, device=prototype_vectors.device)

        loss = 0
        for i in range(num_prototypes):
            pos_indices = torch.where(same_class_mask[i] > 0)[0]
            if len(pos_indices) == 0: continue
            pos_sim = torch.exp(sim_matrix[i, pos_indices])
            neg_indices = torch.where(diff_class_mask[i] > 0)[0]
            neg_sim = torch.sum(torch.exp(sim_matrix[i, neg_indices]))
            curr_loss = -torch.log(pos_sim.sum() / (pos_sim.sum() + neg_sim + 1e-8))
            loss += curr_loss
        return loss / num_prototypes

    def dynamic_cross_entropy_loss(self, p1, p2, labels):
        labels = labels.long()
        pseudo_labels1 = p1.argmax(dim=1)
        pseudo_labels2 = p2.argmax(dim=1)
        consistent_mask = (pseudo_labels1 == pseudo_labels2)
        if not consistent_mask.any():
            return F.cross_entropy(p1, labels)
        return F.cross_entropy(p1[consistent_mask], labels[consistent_mask])

    def get_prototypes(self):
        prototype_vectors = self.task.model.get_prototype_vectors()
        prototypes_by_class = {}
        for class_idx in range(self.num_classes):
            start_idx = class_idx * self.num_prototypes_per_class
            end_idx = (class_idx + 1) * self.num_prototypes_per_class
            prototypes_by_class[class_idx] = prototype_vectors[start_idx:end_idx].detach().cpu()
        return prototypes_by_class