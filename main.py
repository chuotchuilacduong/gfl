from flcore.trainer import FGLTrainer
from utils.basic_utils import seed_everything
import sys

# sys.argv = ["main.py",
#             '--debug', 'False',
#             "--seed", '2024',
#             "--gpuid", '0',
#             '--scenario', 'subgraph_fl',
#             '--simulation_mode', 'subgraph_fl_label_skew',
#             '--task', 'node_cls',
#             "--lr", "0.001",
#             '--louvain_resolution', '1',
#             '--dataset', 'Actor',
#             '--model', 'gcn',
#             '--fl_algorithm', 'fedpub',
#             '--num_clients', '5',
#             '--num_epochs', '3',
#             '--num_rounds','100',
#             '--metrics', 'accuracy',
#             '--evaluation_mode', "local_model_on_local_data"]


from config import args
import numpy as np
if args.seed != 0:
    seed_everything(args.seed)

print(args)

trainer = FGLTrainer(args)
trainer.train()

