import torch
import copy
import numpy as np
import quadprog
from torch.nn.utils import vector_to_parameters
from torch.optim.lr_scheduler import StepLR
from flcore.base import BaseServer

class FedOMGServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device, personalized=False):
        super().__init__(args, global_data, data_dir, message_pool, device, personalized)

    def execute(self):

        sampled_clients = self.message_pool.get("sampled_clients", [])
        if not sampled_clients: return

        inner_models = []
        for cid in sampled_clients:
            if cid in self.message_pool:
                tmp_model = copy.deepcopy(self.task.model)
                tmp_model.load_state_dict(self.message_pool[cid])
                inner_models.append(tmp_model)

        updated_params = self.omg_high(
            meta_weights=self.task.model,
            inner_weights=inner_models,
            lr_meta=self.args.omg_meta_lr
        )
        
        self.task.model.load_state_dict(updated_params)
        
        for cid in sampled_clients:
            if cid in self.message_pool: del self.message_pool[cid]

    def omg_high(self, meta_weights, inner_weights, lr_meta):
        all_domain_grads = []
        flatten_meta = torch.cat([p.data.view(-1) for p in meta_weights.parameters()])
        
        for client_model in inner_weights:
            diffs = [torch.flatten(c_p.data - m_p.data) 
                     for c_p, m_p in zip(client_model.parameters(), meta_weights.parameters())]
            all_domain_grads.append(torch.cat(diffs))

        all_domains_grad_tensor = torch.stack(all_domain_grads).t()

        if self.args.gm_flag == "ca_grad":
            g = self.omg_low(all_domains_grad_tensor, len(inner_weights))
        else:
            g = self.project2cone2(all_domains_grad_tensor, len(inner_weights))

        new_params = flatten_meta + g.to(self.device) * lr_meta
        vector_to_parameters(new_params, meta_weights.parameters())
        
        return meta_weights.state_dict()

    def omg_low(self, grad_vec, num_tasks):
        grads = grad_vec.to(self.device)
        GG = grads.t().mm(grads)
        scale = (torch.diag(GG) + 1e-4).sqrt().mean()
        GG = GG / scale.pow(2)
        Gg = GG.mean(1, keepdims=True)
        
        w = torch.zeros(num_tasks, 1, requires_grad=True, device=self.device)
        w_opt = torch.optim.SGD([w], lr=self.args.omg_learning_rate, momentum=self.args.omg_momentum)
        scheduler = StepLR(w_opt, step_size=self.args.omg_step_size, gamma=self.args.omg_gamma)
        
        c = (Gg.mean(0, keepdims=True) + 1e-4).sqrt() * self.args.omg_c
        w_best = w.clone()
        obj_best = np.inf
        
        for i in range(self.args.omg_rounds):
            w_opt.zero_grad()
            ww = torch.softmax(w, dim=0)
            obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
            if obj.item() < obj_best:
                obj_best = obj.item()
                w_best = w.clone()
            obj.backward()
            w_opt.step()
            scheduler.step()

        ww = torch.softmax(w_best, dim=0)
        gw_norm = (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
        lmbda = c.view(-1) / (gw_norm + 1e-4)
        g = ((1 / num_tasks + ww * lmbda).view(-1, 1) * grads.t()).sum(0) / (1 + self.args.omg_c ** 2)
        return g

    def project2cone2(self, gradient, domain_num, margin=0.5, eps=1e-3):
        memories_np = gradient.cpu().t().detach().double().numpy()
        gradient_np = gradient.mean(1).cpu().view(-1).detach().double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + margin
        v = quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        return torch.Tensor(x).view(-1)

    def send_message(self):
        self.message_pool["server_model"] = copy.deepcopy(self.task.model.state_dict())