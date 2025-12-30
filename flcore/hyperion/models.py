import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential

class PrototypeLayer(nn.Module):
    def __init__(self, output_dim, num_prototypes_per_class, prototype_dim=32, single_target=False):
        super(PrototypeLayer, self).__init__()
        self.output_dim = output_dim
        self.num_prototypes_per_class = num_prototypes_per_class
        self.prototype_dim = prototype_dim
        self.prototype_shape = (output_dim * num_prototypes_per_class, prototype_dim)
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        self.num_prototypes = self.prototype_shape[0]
        self.single_target = single_target
        self.temperature = 0.07

        indices = torch.arange(self.num_prototypes)
        one_hot_vectors = F.one_hot(indices // num_prototypes_per_class, num_classes=output_dim)
        self.register_buffer('prototype_class_identity', one_hot_vectors.float())
    
    def prototype_distances(self, x):
        xp = torch.mm(x, torch.t(self.prototype_vectors))
        distance = -2 * xp + torch.sum(x ** 2, dim=1, keepdim=True) + torch.t(
            torch.sum(self.prototype_vectors ** 2, dim=1, keepdim=True)) 
        return distance

    def proto_nce_loss(self, distances, proto_labels):
        indices = torch.arange(0, self.num_prototypes, device=distances.device)
        class_indices = proto_labels // self.num_prototypes_per_class * self.num_prototypes_per_class
        pos_mask = (indices >= class_indices[:, None]) & (indices < class_indices[:, None] + self.num_prototypes_per_class)
        
        logits = torch.full_like(distances, -1e6)
        logits[pos_mask] = -distances[pos_mask] / self.temperature
        
        targets = (proto_labels // self.num_prototypes_per_class * self.num_prototypes_per_class).to(distances.device)
        return F.cross_entropy(logits, targets)

    def forward(self, x):
        distances = self.prototype_distances(x)
        if not self.single_target:
            _, nearest_prototype_indices = torch.min(distances, dim=1)
            virtual_label = self.prototype_class_identity[nearest_prototype_indices].to(distances.device)
            prot_nce_loss = self.proto_nce_loss(distances, nearest_prototype_indices)
        else:
            virtual_label = None
            prot_nce_loss = None
        return distances, virtual_label, prot_nce_loss

class HyperionModelWrapper(nn.Module):
    def __init__(self, base_gnn_model, args):
        super(HyperionModelWrapper, self).__init__()
        self.gnn = base_gnn_model
        self.prot_dim = args.prot_dim
        self.output_dim = args.num_classes
        
        self.num_prototypes_per_class = args.num_prototypes_per_class
        self.enable_prot = True 

        gnn_out_dim = args.hid_dim 
        
        self.mlp = Sequential(
            Linear(args.hid_dim, self.prot_dim), 
            ReLU(),
            Linear(self.prot_dim, self.output_dim)
        )
        
        self.prototype_layer = PrototypeLayer(
            self.output_dim, 
            self.num_prototypes_per_class,
            prototype_dim=gnn_out_dim, # Prototype
            single_target=args.single_target if hasattr(args, 'single_target') else False
        )

    def get_prototype_vectors(self):
        return self.prototype_layer.prototype_vectors
    
    def get_prototype_class_identity(self):
        return self.prototype_layer.prototype_class_identity

    def forward(self, data):
        out = self.gnn(data)
        if isinstance(out, tuple):
            graph_emb = out[0] 
        else:
            graph_emb = out       
        pred = self.mlp(graph_emb)

        distances, virtual_label, prot_nce_loss = self.prototype_layer(graph_emb)
        return pred, virtual_label, prot_nce_loss, graph_emb, distances