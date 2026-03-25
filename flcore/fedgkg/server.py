import torch
from flcore.fedrgd.server import FedRGDServer
from flcore.fedgm.fedgm_config import config

class FedGKGServer(FedRGDServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedGKGServer, self).__init__(args, global_data, data_dir, message_pool, device)
        
        if not hasattr(self.args, 'method'):
             self.args.method = config.get('method', 'GCond')
        
        print(f"[FedGKG Server] Initialized. Using Condensed Graph for Loss Guidance.")