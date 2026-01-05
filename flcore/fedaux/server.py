import time
import numpy as np
import torch
import networkx as nx
from scipy.spatial.distance import cosine
from flcore.base import BaseServer
from flcore.fedaux.models import GNNAUX
from flcore.fedaux.utils import from_networkx # Đảm bảo hàm này có trong utils hoặc import từ torch_geometric

class FedAuxServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedAuxServer, self).__init__(args, global_data, data_dir, message_pool, device)
        
        # [Original Logic] Khởi tạo các biến theo dõi lịch sử như code gốc
        self.update_lists = []
        self.sim_matrices = []

        # Đảm bảo model là GNNAUX (nếu chưa được load đúng ở task)
        if global_data is not None:
            n_feat = global_data.num_features
            n_dims = args.hid_dim
            n_clss = args.num_classes
            self.task.model = GNNAUX(n_feat, n_dims, n_clss, args).to(device)

        # [Original Logic] Hàm get_proxy_data (giữ nguyên dù chưa thấy dùng trong hàm update)
        # Nếu cần dùng để init model hoặc data, bạn có thể gọi ở đây
        # self.proxy_data = self.get_proxy_data(n_feat)

    def get_proxy_data(self, n_feat):
        # [Original Logic] Giữ nguyên hàm tạo proxy data
        num_graphs, num_nodes = getattr(self.args, 'n_proxy', 5), 100 # Default 5 nếu không có args
        data = from_networkx(nx.random_partition_graph([num_nodes] * num_graphs, p_in=0.1, p_out=0, seed=self.args.seed))
        data.x = torch.normal(mean=0, std=1, size=(num_nodes * num_graphs, n_feat))
        return data

    def execute(self):
        """
        Phương thức này tương đương với hàm 'update' trong code gốc của FedAux.
        Nó thực hiện: Aggregation -> Similarity Matrix -> Personalized Update.
        """
        sampled_clients = self.message_pool["sampled_clients"]
        if not sampled_clients:
            return

        # 1. [Original Logic] Thu thập dữ liệu từ clients (tương ứng đoạn lấy từ self.sd)
        local_weights = []
        local_auxes = []
        local_train_sizes = []
        
        for c_id in sampled_clients:
            msg = self.message_pool[f"client_{c_id}"]
            local_weights.append(msg["weight"])
            # aux cần chuyển sang numpy để tính cosine
            local_auxes.append(msg["aux"].cpu().numpy()) 
            local_train_sizes.append(msg["num_samples"])
        
        # 2. [Original Logic] Tính Similarity Matrix
        st = time.time()
        n_connected = len(sampled_clients)
        sim_matrix = np.empty(shape=(n_connected, n_connected))
        
        for i in range(n_connected):
            for j in range(n_connected):
                # Code gốc: sim_matrix[i, j] = 1 - cosine(...)
                sim_matrix[i, j] = 1 - cosine(local_auxes[i], local_auxes[j])

        # 3. [Original Logic] Normalization (Exp Scaling)
        agg_norm = getattr(self.args, 'agg_norm', 'exp')
        norm_scale = getattr(self.args, 'norm_scale', 10.0) # Default nếu config thiếu
        
        if agg_norm == 'exp':
            sim_matrix = np.exp(norm_scale * sim_matrix)
        
        row_sums = sim_matrix.sum(axis=1)
        sim_matrix = sim_matrix / row_sums[:, np.newaxis]

        # 4. [Original Logic] Update Global Model
        # Code gốc: self.set_weights(self.model, self.aggregate(local_weights, ratio))
        st = time.time()
        ratio = (np.array(local_train_sizes) / np.sum(local_train_sizes)).tolist()
        
        # Gọi hàm helper _aggregate (tự viết để thay thế hàm aggregate của class gốc)
        global_new_weights = self._aggregate(local_weights, ratio)
        self.task.model.load_state_dict(global_new_weights)
        
        # print(f'global model has been updated ({time.time()-st:.2f}s)')

        # 5. [Original Logic] Update Personalized Models
        # Code gốc: duyệt qua clients, aggregate dựa trên sim_matrix row, lưu vào personalized_{c_id}
        st = time.time()
        for i, c_id in enumerate(sampled_clients):
            # Lấy hàng thứ i của sim_matrix làm trọng số
            weights_for_client = sim_matrix[i, :]
            
            # Tính personalized weights
            aggr_local_model_weights = self._aggregate(local_weights, weights_for_client)
            
            # Lưu vào message_pool để dùng cho hàm send_message()
            # (Tương đương việc lưu vào self.sd[f'personalized_{c_id}'])
            self.message_pool[f"server_{c_id}"] = {
                "weight": aggr_local_model_weights
            }
            
        # [Original Logic] Lưu lịch sử
        self.update_lists.append(sampled_clients)
        self.sim_matrices.append(sim_matrix)
        # print(f'local model has been updated ({time.time()-st:.2f}s)')

    def send_message(self):
        """
        Phân phối weights về cho client.
        Tương ứng với việc client lấy 'personalized_{c_id}' hoặc 'global' từ self.sd.
        """
        sampled_clients = self.message_pool.get("sampled_clients", [])
        
        for client_id in sampled_clients:
            # Nếu có personalized weight (đã tính ở execute), gửi nó
            if f"server_{client_id}" in self.message_pool:
                continue # Client sẽ tự động nhận message này
            else:
                # Nếu là vòng đầu (chưa có execute), gửi global weights
                self.message_pool[f"server_{client_id}"] = {
                    "weight": self.task.model.state_dict()
                }

    def _aggregate(self, local_weights, coefficients):
        """
        Hàm helper thực hiện weighted sum các state_dict.
        """
        out_weights = {}
        # Lấy keys từ model đầu tiên
        first_w = local_weights[0]
        
        # Khởi tạo weight rỗng
        for key in first_w.keys():
            out_weights[key] = torch.zeros_like(first_w[key])
            
        # Cộng dồn
        for i, w_dict in enumerate(local_weights):
            coeff = coefficients[i]
            for key in out_weights.keys():
                out_weights[key] += coeff * w_dict[key]
        
        return out_weights