# FedGM/model/fedigl_gin.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, MessagePassing
from torch_geometric.utils import add_self_loops,softmax

class WeightedGINConv(MessagePassing):
    def __init__(self, nn, eps=0, train_eps=False, **kwargs):
        super(WeightedGINConv, self).__init__(aggr='add', **kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=x.dtype, device=x.device)
        edge_index, edge_weight = add_self_loops(edge_index, edge_attr=edge_weight, fill_value=1, num_nodes=x.size(0)) # Fixed num_nodes access
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out += (1 + self.eps) * x
        return self.nn(out)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

class FedIGL_GIN(torch.nn.Module):
    # Copy class GIN from original models.py but renamed to FedIGL_GIN for clarity
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout, args):
        super(FedIGL_GIN, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout

        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))

        self.inv_generator = torch.nn.Sequential(torch.nn.Linear(nhid, nhid))

        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(
            nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
        self.graph_convs.append(WeightedGINConv(self.nn1))
        for l in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(
                nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.graph_convs.append(WeightedGINConv(self.nnk))

        self.graph_convs2 = torch.nn.ModuleList()
        self.nn1_2 = torch.nn.Sequential(torch.nn.Linear(
            nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
        self.graph_convs2.append(WeightedGINConv(self.nn1_2))
        for l in range(nlayer - 1):
            self.nnk_2 = torch.nn.Sequential(torch.nn.Linear(
                nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.graph_convs2.append(WeightedGINConv(self.nnk_2))

        self.graph_convs3 = torch.nn.ModuleList()
        self.nn1_3 = torch.nn.Sequential(torch.nn.Linear(
            nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
        self.graph_convs3.append(WeightedGINConv(self.nn1_3))
        for l in range(nlayer - 1):
            self.nnk_3 = torch.nn.Sequential(torch.nn.Linear(
                nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.graph_convs3.append(WeightedGINConv(self.nnk_3))

        self.graph_convs4 = torch.nn.ModuleList()
        self.nn1_4 = torch.nn.Sequential(torch.nn.Linear(
            nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
        self.graph_convs4.append(WeightedGINConv(self.nn1_4))
        for l in range(nlayer - 1):
            self.nnk_4 = torch.nn.Sequential(torch.nn.Linear(
                nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.graph_convs4.append(WeightedGINConv(self.nnk_4))

        self.post = torch.nn.Sequential(
            torch.nn.Linear(nhid, nhid), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))

        self.prev_grad_1 = None
        self.prev_grad_2 = None

        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3
        self.subgraph_ration = args.subgraph_ration
    def forward_global(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.pre(x)

        for i in range(len(self.graph_convs)):
            inv_x = self.graph_convs3[i](x, edge_index)
            inv_x = F.relu(inv_x)
            inv_x = F.dropout(inv_x, self.dropout, training=self.training)

        m = inv_x @ inv_x.T
        row, col = edge_index
        edge_weight = m[row, col]

        num_edges = edge_weight.size(0)
        num_top_edges = int(self.subgraph_ration * num_edges)
        edge_weight_soft = softmax(edge_weight, row, num_nodes=x.size(0))
        top_edge_indices = torch.topk(edge_weight, num_top_edges)[1]


        top_mask = torch.zeros(
            num_edges, dtype=torch.bool, device=edge_index.device)
        top_mask[top_edge_indices] = True
        remaining_mask = ~top_mask

        top_row = row[top_mask]
        top_col = col[top_mask]
        top_edge_index = torch.stack((top_row, top_col), dim=0)
        top_edge_weight = edge_weight[top_mask]

        remaining_row = row[remaining_mask]
        remaining_col = col[remaining_mask]
        remaining_edge_index = torch.stack(
            (remaining_row, remaining_col), dim=0)
        remaining_edge_weight = edge_weight[remaining_mask]

        for i in range(len(self.graph_convs)):
            x1 = self.graph_convs[i](x, top_edge_index, top_edge_weight)
            x1 = F.relu(x1)
            x1 = F.dropout(x1, self.dropout, training=self.training)

        for i in range(len(self.graph_convs)):
            x2 = self.graph_convs4[i](
                x, remaining_edge_index, remaining_edge_weight)
            x2 = F.relu(x2)
            x2 = F.dropout(x2, self.dropout, training=self.training)

        x5 = global_add_pool(x1 + x2, batch)
        x5 = self.post(x5)
        x5 = F.dropout(x5, self.dropout, training=self.training)
        x5 = self.readout(x5)
        x5 = F.log_softmax(x5, dim=1)

        return x5, x, x1, remaining_edge_index, remaining_edge_weight, batch
        
    def forward_client(self, x, x1, remaining_edge_index, remaining_edge_weight, batch):

        for i in range(len(self.graph_convs)):
            x3 = self.graph_convs2[i](
                x.detach(), remaining_edge_index.detach())
            x3 = F.relu(x3)
            x3 = F.dropout(x3, self.dropout, training=self.training)

        x4 = global_add_pool(x1 + x3, batch)
        x4 = self.post(x4)
        x4 = F.dropout(x4, self.dropout, training=self.training)
        x4 = self.readout(x4)
        x4 = F.log_softmax(x4, dim=1)

        return x4

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

    def backward(self, loss):
        if self.prev_grad_1 is None:
            loss.backward()
            return

        params_1 = list(self.graph_convs.parameters())
        params_2 = list(self.graph_convs4.parameters())
        
        grads_1 = torch.autograd.grad(loss, params_1, create_graph=True, allow_unused=True)
        grads_2 = torch.autograd.grad(loss, params_2, create_graph=True, allow_unused=True)
        
        reg_loss = torch.tensor(0.0, device=loss.device)
        
        if self.prev_grad_1 is not None:
            for g, g_prev in zip(grads_1, self.prev_grad_1):
                if g is not None and g_prev is not None:
                    if g_prev.device != g.device:
                        g_prev = g_prev.to(g.device)
                    reg_loss = reg_loss + self.lambda1 * torch.norm(g - g_prev, p=2)

        if self.prev_grad_2 is not None:
            for g, g_prev in zip(grads_2, self.prev_grad_2):
                if g is not None and g_prev is not None:
                    if g_prev.device != g.device:
                        g_prev = g_prev.to(g.device)
                    reg_loss = reg_loss - self.lambda2 * torch.norm(g - g_prev, p=2)

        total_loss = loss + self.lambda3 * reg_loss

        total_loss.backward()
class FedIGL_Node_GIN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout, args):
        super(FedIGL_Node_GIN, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout

        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))

        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
        self.graph_convs.append(WeightedGINConv(self.nn1))
        for l in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.graph_convs.append(WeightedGINConv(self.nnk))

        self.graph_convs2 = torch.nn.ModuleList()
        self.nn1_2 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
        self.graph_convs2.append(WeightedGINConv(self.nn1_2))
        for l in range(nlayer - 1):
            self.nnk_2 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.graph_convs2.append(WeightedGINConv(self.nnk_2))

        self.graph_convs3 = torch.nn.ModuleList()
        self.nn1_3 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
        self.graph_convs3.append(WeightedGINConv(self.nn1_3))
        for l in range(nlayer - 1):
            self.nnk_3 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.graph_convs3.append(WeightedGINConv(self.nnk_3))

        self.graph_convs4 = torch.nn.ModuleList()
        self.nn1_4 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
        self.graph_convs4.append(WeightedGINConv(self.nn1_4))
        for l in range(nlayer - 1):
            self.nnk_4 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.graph_convs4.append(WeightedGINConv(self.nnk_4))

        self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))

        self.prev_grad_1 = None
        self.prev_grad_2 = None
        self.stored_grads_1 = None
        self.stored_grads_2 = None
        self.lambda1 = getattr(args, 'lambda1', 1.0)
        self.lambda2 = getattr(args, 'lambda2', 1.0)
        self.lambda3 = getattr(args, 'lambda3', 0.1)
        self.subgraph_ration = getattr(args, 'subgraph_ration', 0.5)

    def forward(self, data):
        """
        Standard forward pass for evaluation.
        Delegates to forward_global logic but returns format expected by NodeClsTask.
        """
        # Gọi forward_global để lấy logits (x5)
        # forward_global trả về: x5, x, x1, remaining_edge_index, remaining_edge_weight
        logits, _, _, _, _ = self.forward_global(data)
        
        # NodeClsTask mong đợi output là (embedding, logits)
        # Ở đây ta trả về (logits, logits) vì default_loss_fn chỉ dùng logits, 
        # và ta không cần embedding riêng cho logic này.
        return logits, logits

    def forward_global(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.pre(x)

        for i in range(len(self.graph_convs)):
            inv_x = self.graph_convs3[i](x, edge_index)
            inv_x = F.relu(inv_x)
            inv_x = F.dropout(inv_x, self.dropout, training=self.training)

        m = inv_x @ inv_x.T
        row, col = edge_index
        edge_weight = m[row, col]

        num_edges = edge_weight.size(0)
        num_top_edges = int(self.subgraph_ration * num_edges)
        edge_weight_soft = softmax(edge_weight, row, num_nodes=x.size(0))
        
        # Fix: clamp num_top_edges to avoid error if num_edges is small
        k = min(num_top_edges, num_edges)
        if k > 0:
            top_edge_indices = torch.topk(edge_weight_soft, k)[1]
        else:
            top_edge_indices = torch.tensor([], dtype=torch.long, device=edge_index.device)

        top_mask = torch.zeros(num_edges, dtype=torch.bool, device=edge_index.device)
        top_mask[top_edge_indices] = True
        remaining_mask = ~top_mask

        top_edge_index = edge_index[:, top_mask]
        top_edge_weight = edge_weight_soft[top_mask]

        remaining_edge_index = edge_index[:, remaining_mask]
        remaining_edge_weight = edge_weight_soft[remaining_mask]

        for i in range(len(self.graph_convs)):
            x1 = self.graph_convs[i](x, top_edge_index, top_edge_weight)
            x1 = F.relu(x1)
            x1 = F.dropout(x1, self.dropout, training=self.training)

        for i in range(len(self.graph_convs)):
            x2 = self.graph_convs4[i](x, remaining_edge_index, remaining_edge_weight)
            x2 = F.relu(x2)
            x2 = F.dropout(x2, self.dropout, training=self.training)

        x5 = x1 + x2 
        x5 = self.post(x5)
        x5 = F.dropout(x5, self.dropout, training=self.training)
        x5 = self.readout(x5)
        
        return x5, x, x1, remaining_edge_index, remaining_edge_weight

    def forward_client(self, x, x1, remaining_edge_index, remaining_edge_weight):
        for i in range(len(self.graph_convs)):
            x3 = self.graph_convs2[i](x, remaining_edge_index, remaining_edge_weight)
            x3 = F.relu(x3)
            x3 = F.dropout(x3, self.dropout, training=self.training)

        x4 = x1 + x3
        x4 = self.post(x4)
        x4 = F.dropout(x4, self.dropout, training=self.training)
        x4 = self.readout(x4)

        return x4

    def loss(self, pred, label):
        return F.cross_entropy(pred, label)

    def backward(self, loss):
        if self.prev_grad_1 is None:
            loss.backward()
            return

        params_1 = list(self.graph_convs.parameters())
        params_2 = list(self.graph_convs4.parameters())
        
        grads_1 = torch.autograd.grad(loss, params_1, create_graph=True, allow_unused=True)
        grads_2 = torch.autograd.grad(loss, params_2, create_graph=True, allow_unused=True)
        
        self.stored_grads_1 = [g.detach().clone() if g is not None else torch.zeros_like(p) 
                               for g, p in zip(grads_1, params_1)]
        self.stored_grads_2 = [g.detach().clone() if g is not None else torch.zeros_like(p) 
                               for g, p in zip(grads_2, params_2)]
        reg_loss = torch.tensor(0.0, device=loss.device)
        
        if self.prev_grad_1 is not None:
            for g, g_prev in zip(grads_1, self.prev_grad_1):
                if g is not None and g_prev is not None:
                    if g_prev.device != g.device:
                        g_prev = g_prev.to(g.device)
                    reg_loss = reg_loss + self.lambda1 * torch.norm(g - g_prev, p=2)

        if self.prev_grad_2 is not None:
            for g, g_prev in zip(grads_2, self.prev_grad_2):
                if g is not None and g_prev is not None:
                    if g_prev.device != g.device:
                        g_prev = g_prev.to(g.device)
                    reg_loss = reg_loss - self.lambda2 * torch.norm(g - g_prev, p=2)

        total_loss = loss + self.lambda3 * reg_loss
        total_loss.backward()