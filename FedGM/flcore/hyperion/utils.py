import torch
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture

# --- Các hàm hỗ trợ tính toán khoảng cách và Sinkhorn ---

def sinkhorn(w1, w2, cost, reg=0.05, max_iter=10):
    bs, dim = w1.shape
    w1 = w1.unsqueeze(-1)
    w2 = w2.unsqueeze(-1)

    u = 1/dim*torch.ones_like(w1, device=w1.device, dtype=w1.dtype) 
    K = torch.exp(-cost / reg)
    Kt= K.transpose(2, 1)
    for i in range(max_iter):
        v=w2/(torch.bmm(Kt,u)+1e-8) 
        u=w1/(torch.bmm(K,v)+1e-8) 

    flow = u.reshape(bs, -1, 1) * K * v.reshape(bs, 1, -1)
    return flow

def wkd_prototype_loss(logits_student, logits_teacher, temperature, gamma, cost_matrix=None, sinkhorn_lambda=0.05, sinkhorn_iter=10):
    if logits_student.dim() == 1:
        logits_student = logits_student.unsqueeze(0)  
    if logits_teacher.dim() == 1:
        logits_teacher = logits_teacher.unsqueeze(0) 

    logits_student = logits_student.to(torch.float32)
    logits_teacher = logits_teacher.to(torch.float32)

    pred_student = F.softmax(logits_student / temperature, dim=-1).to(torch.float32)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=-1).to(torch.float32)

    cost_matrix = F.relu(cost_matrix) + 1e-8
    if cost_matrix.dim() == 2:
        cost_matrix = cost_matrix.unsqueeze(0)
    cost_matrix = cost_matrix.to(torch.float32).to(pred_student.device)
    
    flow = sinkhorn(pred_student, pred_teacher, cost_matrix, reg=sinkhorn_lambda, max_iter=sinkhorn_iter)

    ws_distance = (flow * cost_matrix).sum(-1).sum(-1)
    ws_distance = ws_distance.mean()
    return gamma * ws_distance

def calc_cost_matrix_proto(initial_global_prototypes, num_prototypes_per_class, device):
    # Chuyển prototypes từ dict sang tensor
    prototypes_tensor = torch.stack(
        [proto.to(device) for proto in initial_global_prototypes.values()]
    )  
    prototypes_flat = prototypes_tensor.reshape(-1, prototypes_tensor.shape[-1]) 
    
    def stable_cosine_distance(x):
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-8)  
        sim_matrix = torch.mm(x_norm, x_norm.T)       
        return 1 - sim_matrix                        
    
    cost_matrix = stable_cosine_distance(prototypes_flat.T)  
    cost_matrix.fill_diagonal_(0.0)  
    cost_matrix = torch.clamp(cost_matrix, min=0, max=2)  
    return cost_matrix

def get_node_prototype_bool_matrix(data, model, device):
    node_labels = data.y.to(device)  
    prototype_class_identity = model.get_prototype_class_identity().to(device)  
    node_prototype_bool = prototype_class_identity[:, node_labels].bool().t()  
    return node_prototype_bool

def prune_training_set(data, model, distances, retain_ratio, device):
    # Logic cắt tỉa dữ liệu (pruning)
    train_indices = torch.where(data.train_mask)[0]
    node_prototype_bool = get_node_prototype_bool_matrix(data, model, device)
    
    min_distances = torch.zeros(distances.shape[0], device=device)
    for i in range(distances.shape[0]):
        class_proto_dist = distances[i][node_prototype_bool[i]]
        if class_proto_dist.numel() > 0:
            min_distances[i] = torch.min(class_proto_dist)
        else:
            min_distances[i] = 1000.0
    
    train_min_distances = min_distances[train_indices]
    sorted_train_indices = torch.argsort(train_min_distances) 
    retain_count = max(1, int(len(train_min_distances) * retain_ratio))  
    top_train_indices = sorted_train_indices[:retain_count]
    retained_train_indices = train_indices[top_train_indices]

    new_train_mask = torch.zeros_like(data.train_mask, dtype=torch.bool)
    new_train_mask[retained_train_indices] = True
    data.train_mask = new_train_mask
    return data