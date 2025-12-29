from flcore.trainer import FGLTrainer
from utils.basic_utils import seed_everything
import sys

sys.argv = ["main.py",
            '--debug', 'False',
            "--seed", '2024',
            "--gpuid", '0',
            '--scenario', 'subgraph_fl',
            '--simulation_mode', 'subgraph_fl_label_skew',
            '--task', 'node_cls',
            "--lr", "0.005",
            '--louvain_resolution', '1',
            '--dataset', 'Cora',
            '--model', 'gcn',
            '--fl_algorithm', 'fedlog',
            '--num_clients', '10',
            '--num_epochs', '2',
            '--num_rounds','100',
            '--metrics', 'accuracy',
            '--evaluation_mode', "local_model_on_local_data"]


from config import args
import numpy as np
if args.seed != 0:
    seed_everything(args.seed)

print(args)

trainer = FGLTrainer(args)
trainer.train()

