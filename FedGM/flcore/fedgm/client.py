import torch
import torch.nn as nn
from flcore.base import BaseClient
from flcore.fedgm.fedgm_config import config
from flcore.fedgm.pge import PGE
from flcore.fedgm.IGNR import GraphonLearner as IGNR
from torch_geometric.utils.sparse import to_torch_sparse_tensor
import random
import numpy as np
from torch_sparse import SparseTensor
from flcore.fedgm.utils import match_loss, regularization, tensor2onehot
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler
import numpy as np
from data.simulation import get_subgraph_pyg_data
from torch_geometric.data import Data
from torch_geometric.utils import to_edge_index
from flcore.fedgm.utils import is_sparse_tensor, normalize_adj_tensor, accuracy
from model.gcn import GCN_kipf
from utils.metrics import compute_supervised_metrics
import copy


class FedGMClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedGMClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.class_dict2 = None
        self.samplers = None
        
        if not hasattr(self.args, 'method'):
            self.args.method = 'GCond'
        
        self.model = GCN_kipf(nfeat=self.task.num_feats, 
                                             nhid=args.hid_dim, 
                                             nclass=self.task.num_global_classes, 
                                             nlayers=args.num_layers, 
                                             dropout=args.dropout, 
                                             lr=args.lr, 
                                             weight_decay=args.weight_decay, 
                                             device=self.device).to(self.device)
        print(f"client {self.client_id} graph condensation")

        self.opt_epochs = 0
        self.fedgc_initialization()
        self.task.override_evaluate = self.get_override_evaluate()

        # Select real nodes as the feature matrix.
        self.syn_x.data.copy_(self.get_syn_x_initialize(self.task.splitted_data))

        self.best_loss = float('inf')
        self.best_syn_x = None
        self.best_pge_state = None

        for i in range(config['op_epoche']):
            self.model.initialize()
            self.get_gradient(self.task.splitted_data)
            self.global_class_gradient = {}
            self.gradient_match(self.task.splitted_data, self.global_class_gradient, is_global = False)



    def execute(self):
        if self.message_pool["round"] != 0:
            with torch.no_grad():
                for (local_param, global_param) in zip(self.model.parameters(), self.message_pool["server"]["weight"]):
                    local_param.data.copy_(global_param)
            self.get_gradient(self.task.splitted_data)
        return
        

    def send_message(self):
        if self.message_pool["round"] == 0:
            self.message_pool[f"client_{self.client_id}"]={
                "syn_x": self.best_syn_x,
                "syn_y": self.syn_y,
                "pge": self.best_pge_state,
                "num_syn_nodes": self.num_syn_nodes,
                "local_syn_class_indices": self.syn_class_indices,
                "method": self.args.method
            }
        else:
            self.message_pool[f"client_{self.client_id}"]={
                "local_class_gradient":self.local_class_gradient,
                "local_num_class_dict":self.num_class_dict,
        }

        
    def fedgc_initialization(self):
        # conduct local subgraph condensation
        self.num_real_train_nodes = self.task.splitted_data["train_mask"].sum()
        self.num_syn_nodes = int(self.num_real_train_nodes * config["reduction_rate"])
        
        # trainable parameters
        self.syn_x = nn.Parameter(torch.FloatTensor(self.num_syn_nodes, self.task.num_feats).to(self.device))
        if self.args.method == 'SGDD':
            self.pge = IGNR(node_feature=self.task.num_feats, nfeat=128, nnodes=self.num_syn_nodes, 
                            device=self.device, args=self.args).to(self.device)
        else:
            # GCond và DosCond sử dụng PGE
            self.pge = PGE(nfeat=self.task.num_feats, nnodes=self.num_syn_nodes, device=self.device, args=self.args).to(self.device)
        # sampled syn labels
        self.syn_y = torch.LongTensor(self.generate_labels_syn(self.task.splitted_data)).to(self.device)

        # initialize trainable parameters and create optimizer
        self.reset_parameters()
        self.optimizer_feat = torch.optim.Adam([self.syn_x], lr=config["lr_feat"]) # parameterized syn_x
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=config["lr_adj"]) # pge: mapping for syn_x -> syn_adj
        
        # print shape
        print('syn_adj:', (self.num_syn_nodes, self.num_syn_nodes), 'syn_x:', self.syn_x.shape)
    
    
    
    def reset_parameters(self):
        self.syn_x.data.copy_(torch.randn(self.syn_x.size()))

    def generate_labels_syn(self, splitted_data):
        from collections import Counter
        labels_train = splitted_data["data"].y[splitted_data["train_mask"]]
        labels_train = labels_train.tolist()
        counter = Counter(labels_train)
        print(counter)
        num_class_dict = {}
        num_train = self.num_real_train_nodes

        sorted_counter = sorted(counter.items(), key=lambda x:x[1])
        # print(sorted_counter)
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}

        max_class_index = None
        max_class_count = 0

        for c, num in sorted_counter:
            if num > max_class_count:
                max_class_count = num
                max_class_index = c
        
        for ix, (c, num) in enumerate(sorted_counter):
            num_class = int(num * config["reduction_rate"])
            if num_class < 1:
                num_class_dict[c] = 0
                continue 
            num_class_dict[c] = num_class
            sum_ += num_class_dict[c]
            self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
            labels_syn += [c] * num_class_dict[c]

        if len(labels_syn) < int(num_train * config["reduction_rate"]):
            additional_num = int(num_train * config["reduction_rate"]) - len(labels_syn)
            self.syn_class_indices[max_class_index] = [self.syn_class_indices[max_class_index][0], 
                                                    self.syn_class_indices[max_class_index][1] + additional_num]
            labels_syn += [max_class_index] * additional_num
            num_class_dict[max_class_index] += additional_num
    
        self.num_class_dict = num_class_dict
        # print(num_class_dict)
        return labels_syn
    def sampling(self, ids_per_cls_train, budget, vecs, d, using_half=True):
        """
        Hàm sampling được tích hợp từ mã bạn cung cấp.
        Đã điều chỉnh để tương thích với ids_per_cls_train dạng Dictionary.
        """
        budget_dist_compute = 1000
        
        if isinstance(vecs, np.ndarray):
            vecs = torch.from_numpy(vecs)
        
        # Chuyển về half precision nếu cần, nhưng giữ bản gốc để tránh lỗi tính toán nếu logic sau cần float
        vecs_proc = vecs.half() if using_half else vecs

        ids_selected = []
        
        # Lấy danh sách các key (label) để truy cập dictionary an toàn
        cls_keys = list(budget.keys())
        
        for i, class_key in enumerate(cls_keys):
            # class_ = class_key (Label thực tế)
            
            # Tạo danh sách index của các lớp KHÁC lớp hiện tại
            other_cls_indices = list(range(len(cls_keys)))
            other_cls_indices.pop(i)
            
            # Lấy danh sách ID của lớp hiện tại từ Dictionary
            current_cls_ids = ids_per_cls_train[class_key]

            ids_selected0 = (
                current_cls_ids
                if len(current_cls_ids) < budget_dist_compute
                else random.choices(current_cls_ids, k=budget_dist_compute)
            )

            dist = []
            vecs_0 = vecs_proc[ids_selected0]
            
            for j in other_cls_indices:
                other_class_key = cls_keys[j] # Lấy label của lớp khác
                other_cls_ids_list = ids_per_cls_train[other_class_key] # Truy cập dict bằng label
                
                chosen_ids = random.choices(
                    other_cls_ids_list,
                    k=min(budget_dist_compute, len(other_cls_ids_list)),
                )
                vecs_1 = vecs_proc[chosen_ids]
                
                # Tính khoảng cách
                if len(chosen_ids) < 26 or len(ids_selected0) < 26:
                    # torch.cdist throws error for tensor smaller than 26
                    dist.append(torch.cdist(vecs_0.float(), vecs_1.float()).half())
                else:
                    dist.append(torch.cdist(vecs_0, vecs_1))

            dist_ = torch.cat(dist, dim=-1)  # include distance to all the other classes
            n_selected = (dist_ < d).sum(dim=-1)
            rank = n_selected.sort()[1].tolist()
            
            current_ids_selected = (
                rank[: budget[class_key]]
                if len(rank) > budget[class_key]
                else random.choices(rank, k=budget[class_key])
            )
            ids_selected.extend([ids_selected0[j] for j in current_ids_selected]) # Lưu ý: index j lấy từ ids_selected0
            
        return ids_selected
    def get_syn_x_initialize(self, splitted_data):
        if self.args.method == 'SGDD':
            from collections import Counter
            counter = Counter(self.syn_y.cpu().numpy())
            features = splitted_data["data"].x
            labels = splitted_data["data"].y
            train_mask = splitted_data["train_mask"]

            ids_per_cls_train = {}
            for c in counter.keys():
                idx = ((labels == c) & train_mask).nonzero().squeeze().tolist()
                if not isinstance(idx, list):
                    idx = [idx]
                ids_per_cls_train[c] = idx
            d_threshold = getattr(self.args, 'd', 0.5)
            idx_selected = self.sampling(ids_per_cls_train, counter, features, d=d_threshold)
            
            idx_selected = np.array(idx_selected).reshape(-1)
            sub_x = features[idx_selected]
            return sub_x
        idx_selected = []
        from collections import Counter;
        counter = Counter(self.syn_y.cpu().numpy())
        for c in range(self.task.num_global_classes):
            # sample 'counter[c]' nodes in class 'c'
            tmp = ((splitted_data["data"].y == c) & splitted_data["train_mask"]).nonzero().squeeze().tolist()
            if type(tmp) is not list:
                tmp = [tmp]
            if len(tmp) < counter[c]:
                num_additional = counter[c] - len(tmp)
                tmp.extend( random.choices(tmp, k = num_additional))
            random.shuffle(tmp)
            tmp = tmp[:counter[c]]
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        sub_x = splitted_data["data"].x[idx_selected]
        return sub_x

    def get_gradient(self,splitted_data):
        self.local_class_gradient = {}

        real_x, real_adj, real_y = splitted_data["data"].x, to_torch_sparse_tensor(splitted_data["data"].edge_index), splitted_data["data"].y
        if is_sparse_tensor(real_adj):
            real_adj_norm = normalize_adj_tensor(real_adj, sparse=True)
        else:
            real_adj_norm = normalize_adj_tensor(real_adj)
        real_adj_norm = SparseTensor(row=real_adj_norm._indices()[0], col=real_adj_norm._indices()[1],
                value=real_adj_norm._values(), sparse_sizes=real_adj_norm.size()).t()
        
        model_parameters = list(self.model.parameters())
        self.model.train()

        BN_flag = False
        for module in self.model.modules():
            if 'BatchNorm' in module._get_name(): #BatchNorm
                BN_flag = True
        if BN_flag:
            self.model.train() # for updating the mu, sigma of BatchNorm
            output_real = self.model.forward(real_x, real_adj_norm)
            for module in self.model.modules():
                if 'BatchNorm' in module._get_name():  #BatchNorm
                    module.eval() # fix mu and sigma of every BatchNorm layer
        
        # class-wise
        for c in range(self.task.num_global_classes):
            # real loss
            batch_size, n_id, adjs = self.retrieve_class_sampler(c, real_adj_norm, args=self.args)
            if n_id == None:
                continue
            if self.args.num_layers == 1:
                adjs = [adjs]

            adjs = [adj.to(self.device) for adj in adjs]
            output = self.model.forward_sampler(real_x[n_id], adjs)
            
            loss_real = F.nll_loss(output, real_y[n_id[:batch_size]])

            gw_real = torch.autograd.grad(loss_real, model_parameters)
            gw_real = list((_.detach().clone() for _ in gw_real))

            self.local_class_gradient[c] = gw_real 
            
    def gradient_match(self,splitted_data, global_class_gradient,is_global=False):
        syn_x, pge, syn_y = self.syn_x, self.pge, self.syn_y
        syn_class_indices = self.syn_class_indices
        real_x, real_adj, real_y = splitted_data["data"].x, to_torch_sparse_tensor(splitted_data["data"].edge_index), splitted_data["data"].y
        if is_sparse_tensor(real_adj):
            real_adj_norm = normalize_adj_tensor(real_adj, sparse=True)
        else:
            real_adj_norm = normalize_adj_tensor(real_adj)
        real_adj_norm = SparseTensor(row=real_adj_norm._indices()[0], col=real_adj_norm._indices()[1],
                value=real_adj_norm._values(), sparse_sizes=real_adj_norm.size()).t()

        model_parameters = list(self.model.parameters())
        self.model.train()

        if self.args.method == 'SGDD':
            adj_syn, opt_loss = pge(self.syn_x, Lx=None) 
        else:
            adj_syn = pge(self.syn_x)
        adj_syn_norm = normalize_adj_tensor(adj_syn, sparse=False)
        loss = torch.tensor(0.0).to(self.device)
              
        # class-wise
        for c in range(self.task.num_global_classes):
            if c not in self.syn_class_indices:
                continue
            batch_size, n_id, adjs = self.retrieve_class_sampler(c, real_adj_norm, args=self.args)
            if n_id == None:
                continue
            if self.args.num_layers == 1:
                adjs = [adjs]
    
            # syn loss
            output_syn = self.model.forward(syn_x, adj_syn_norm)
            ind = syn_class_indices[c]
            loss_syn = F.nll_loss(
                    output_syn[ind[0]: ind[1]],
                    syn_y[ind[0]: ind[1]])
            gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
            
            # gradient match
            coeff = self.num_class_dict[c] / max(self.num_class_dict.values())

            local_c_loss = match_loss(gw_syn, self.local_class_gradient[c], config["dis_metric"], device=self.device)
            if is_global == False or c not in global_class_gradient:
                loss += coeff  * local_c_loss
            else: 
                global_c_loss = match_loss(gw_syn, global_class_gradient[c], config["dis_metric"], device=self.device)
                loss += coeff  * (config["local_loss_ratio"] * local_c_loss + (1 - config["local_loss_ratio"]) * global_c_loss)
        
        # TODO: regularize
        if config["alpha"] > 0:
            loss_reg = config["alpha"]* regularization(adj_syn, tensor2onehot(syn_y))
        else:
            loss_reg = torch.tensor(0)

        if self.args.method == 'SGDD' and config.get("opt_scale", 0) > 0:
             loss_opt_scaled = config["opt_scale"] * opt_loss
        else:
             loss_opt_scaled = torch.tensor(0.0).to(self.device)
        loss = loss + loss_reg+ loss_opt_scaled

        if loss.item() < self.best_loss:
            self.best_loss = loss.item()
            self.best_syn_x = copy.deepcopy(syn_x)
            self.best_pge_state = copy.deepcopy(list(self.pge.parameters()))

        # update sythetic graph
        self.optimizer_feat.zero_grad()
        self.optimizer_pge.zero_grad()
        loss.backward()
        if is_global == False:
            if self.opt_epochs % (config["opti_pge_epoche"] + config["opti_x_epoche"]) < config["opti_pge_epoche"]:
                self.optimizer_pge.step()
                self.opt_epochs += 1
            else:
                self.optimizer_feat.step()
                self.opt_epochs += 1
        else:
            self.optimizer_feat.step()



        
    def retrieve_class_sampler(self, c, adj, num=256, args=None):

        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.task.num_global_classes):
                idx_mask = (self.task.splitted_data["data"].y == i) & self.task.splitted_data["train_mask"]
                idx = idx_mask.nonzero().squeeze().tolist()
                if type(idx) is not list:
                    idx = [idx]
                self.class_dict2[i] = idx

        if args.num_layers == 1:
            sizes = [15]
        elif args.num_layers == 2:
            sizes = [10, 5]
        elif args.num_layers == 3:
            sizes = [15, 10, 5]
        elif args.num_layers == 4:
            sizes = [15, 10, 5, 5]
        elif args.num_layers == 5:
            sizes = [15, 10, 5, 5, 5]
        else:
            raise ValueError("Unsupported number of layers: {}".format(args.num_layers))

        if self.samplers is None:
            self.samplers = []
            for i in range(self.task.num_global_classes):
                node_idx = torch.LongTensor(self.class_dict2[i])
                valid_node_idx = node_idx[node_idx < adj.size(0)]
                if valid_node_idx.shape[0] == 0:
                    self.samplers.append(None)
                else:
                    self.samplers.append(NeighborSampler(adj,
                                        node_idx=valid_node_idx,
                                        sizes=sizes, batch_size=num,
                                        num_workers=12, return_e_id=False,
                                        num_nodes=adj.size(0),
                                        shuffle=True))

        if len(self.class_dict2[c]) == 0:
            return None, None, None

        valid_class_indices = [idx for idx in self.class_dict2[c] if idx < adj.size(0)]
        if len(valid_class_indices) == 0:
            return None, None, None
        actual_num = min(num, len(valid_class_indices))
        batch = np.random.permutation(valid_class_indices)[:actual_num]
        batch = torch.from_numpy(batch).long()
        if self.samplers[c] is not None:
            try:
                out = self.samplers[c].sample(batch)
                return out
            except IndexError as e:
                print(f"IndexError during sampling: {e}")
                return None, None, None
        else:
            return None, None, None

    
    
    def get_override_evaluate(self):
        def override_evaluate(splitted_data=None, mute=False):
            if splitted_data is None:
                splitted_data = self.task.splitted_data
            else:
                names = ["data", "train_mask", "val_mask", "test_mask"]
                for name in names:
                    assert name in splitted_data
           
            eval_output = {}

            args=self.args
            
            syn_x, pge, syn_y = self.best_syn_x.detach(), self.pge, self.syn_y

            for (local_param, global_param) in zip(self.best_pge_state, pge.parameters()):
                    global_param.data.copy_(local_param)
                    
            if self.args.method == 'SGDD':
                 adj_syn, _ = pge(syn_x, Lx=None) # SGDD inference
            else:
                 adj_syn = pge.inference(syn_x)
            real_adj = to_torch_sparse_tensor(splitted_data["data"].edge_index)
            
            
            # with_bn = True if args.dataset in ['ogbn-arxiv'] else False
            self.task.load_custom_model(GCN_kipf(nfeat=self.task.num_feats, 
                                                nhid=args.hid_dim, 
                                                nclass=self.task.num_global_classes, 
                                                nlayers=args.num_layers, 
                                                dropout=args.dropout, 
                                                lr=args.lr, 
                                                weight_decay=args.weight_decay, 
                                                device=self.device))

                
            self.task.model.initialize()

            if self.args.debug:
                torch.save(adj_syn, f'./saved_ours/adj_{args.dataset}_{config["reduction_rate"]}_{args.seed}.pt')
                torch.save(syn_x, f'./saved_ours/feat_{args.dataset}_{config["reduction_rate"]}_{args.seed}.pt')

            if config["lr_adj"] == 0:
                n = len(syn_y)
                adj_syn = torch.zeros((n, n))

            self.task.model.fit_with_val(syn_x, adj_syn, syn_y, self.task.splitted_data, train_iters=300, normalize=True, verbose=False,adj_val=True)

            self.task.model.eval()
            with torch.no_grad():
                embedding = None
                logits = self.task.model.predict(splitted_data["data"].x, real_adj)
                loss_train = F.nll_loss(logits[splitted_data["train_mask"]],splitted_data["data"].y[splitted_data["train_mask"]])
                loss_val = F.nll_loss(logits[splitted_data["val_mask"]], splitted_data["data"].y[splitted_data["val_mask"]])
                loss_test = F.nll_loss(logits[splitted_data["test_mask"]], splitted_data["data"].y[splitted_data["test_mask"]])

            
            eval_output["embedding"] = embedding
            eval_output["logits"] = logits
            eval_output["loss_train"] = loss_train
            eval_output["loss_val"]   = loss_val
            eval_output["loss_test"]  = loss_test
            
            
            metric_train = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["train_mask"]], labels=splitted_data["data"].y[splitted_data["train_mask"]], suffix="train")
            metric_val = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["val_mask"]], labels=splitted_data["data"].y[splitted_data["val_mask"]], suffix="val")
            metric_test = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["test_mask"]], labels=splitted_data["data"].y[splitted_data["test_mask"]], suffix="test")
            eval_output = {**eval_output, **metric_train, **metric_val, **metric_test}
            
            info = ""
            for key, val in eval_output.items():
                try:
                    info += f"\t{key}: {val:.4f}"
                except:
                    continue

            prefix = f"[client {self.client_id}]" if self.client_id is not None else "[server]"
            if not mute:
                print(prefix+info)
            return eval_output
        
        return override_evaluate