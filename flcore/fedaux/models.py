import torch.nn as nn
from torch.nn import functional as F, init
import math
from utils import *
import torch
class GCN(nn.Module):
    def __init__(self, n_feat=10, n_dims=128, n_clss=10, args=None):
        super().__init__()
        self.n_feat = n_feat
        self.n_dims = n_dims
        self.n_clss = n_clss
        self.args = args

        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(self.n_feat, self.n_dims, cached=False)
        self.conv2 = GCNConv(self.n_dims, self.n_dims, cached=False)
        self.clsif = nn.Linear(self.n_dims, self.n_clss)

    def forward(self, data, is_proxy=False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        if is_proxy == True: return x
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.clsif(x)
        return x

class MaskedGCN(nn.Module):
    def __init__(self, n_feat=10, n_dims=128, n_clss=10, l1=1e-3, args=None):
        super().__init__()
        self.n_feat = n_feat
        self.n_dims = n_dims
        self.n_clss = n_clss
        self.args = args
        
        from layers import MaskedGCNConv, MaskedLinear
        self.conv1 = MaskedGCNConv(self.n_feat, self.n_dims, cached=False, l1=l1, args=args)
        self.conv2 = MaskedGCNConv(self.n_dims, self.n_dims, cached=False, l1=l1, args=args)
        self.clsif = MaskedLinear(self.n_dims, self.n_clss, l1=l1, args=args)

    def forward(self, data, is_proxy=False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        if is_proxy == True: return x
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.clsif(x)
        return x


import torch
import torch.nn as nn
from torch.nn import functional as F, init
from torch_geometric.nn import GCNConv, SAGEConv

class GNNAUX(nn.Module):
    def __init__(self, n_feat, n_dims, n_clss, args=None):
        super().__init__()
        self.conv1 = SAGEConv(n_feat, n_dims)
        self.conv2 = SAGEConv(n_dims, n_dims)
        anchor_dim = n_dims
        self.aux = nn.Parameter(torch.empty(anchor_dim))  # \boldsymbol a
        init.normal_(self.aux)
        self.clsif = nn.Linear(2 * n_dims, n_clss)
        
        self.sigma = args.sigma
        # Sử dụng dropout1 nếu có, nếu không dùng dropout thường
        self.dropout1 = getattr(args, 'dropout1', args.dropout)

    # ---------- kernel‑aggregator ------------------------------------
    def _kernel_aggregate(self, h: torch.Tensor, edge_index) -> torch.Tensor:
        """
        h : [N, d]           node embeddings
        returns z : [N, d]   kernel‑smoothed embeddings
        """
        a = F.normalize(self.aux, dim=0)                          # unit vector
        score = F.cosine_similarity(h, a.unsqueeze(0), dim=-1)    # s_i  (N)
        diff  = score.unsqueeze(0) - score.unsqueeze(1)           # (N,N)
        
        # Lưu ý: Với đồ thị lớn, tính diff (N,N) có thể gây OOM. 
        # FedAux gốc có thể đã được thiết kế cho đồ thị nhỏ hoặc phân vùng.
        # Ở đây giữ nguyên logic, nhưng cần cẩn trọng với bộ nhớ.
        
        kappa = torch.exp(-(diff ** 2) / (self.sigma ** 2))       # eq.(3)
        z     = (kappa @ h) / kappa.sum(dim=1, keepdim=True)
        return z

    # ---------- forward pass -----------------------------------------
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout1, training=self.training)
        h = self.conv2(h, edge_index)
        z = self._kernel_aggregate(h, edge_index)
        
        # Concatenate embedding gốc và smoothed embedding
        combined = torch.cat([h, z], dim=-1)
        out = self.clsif(combined)
        
        # [QUAN TRỌNG] Trả về (embedding, logits) để khớp với NodeClsTask.evaluate
        return h, out