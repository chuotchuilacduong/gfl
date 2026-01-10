import torch
import copy
from flcore.base import BaseClient

class FedOMGClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device, personalized=False):
        super().__init__(args, client_id, data, data_dir, message_pool, device, personalized)

    def execute(self):
        if "server_model" in self.message_pool:
            self.task.model.load_state_dict(self.message_pool["server_model"])
        self.task.train() 
        self.send_message()

    def send_message(self):

        self.message_pool[self.client_id] = copy.deepcopy(self.task.model.state_dict())