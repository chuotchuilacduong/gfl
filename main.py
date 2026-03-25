from flcore.trainer import FGLTrainer
from utils.basic_utils import seed_everything
import sys

# sys.argv hardcoding removed so terminal arguments are respected
from config import args
import numpy as np
fedgm_based_algorithms = ['fedgm', 'fedrgd', 'fedgkg']

if args.fl_algorithm in fedgm_based_algorithms:
    try:
        from flcore.fedgm.fedgm_config import config as fedgm_cfg
        print(f"Loading extra config for {args.fl_algorithm}...")

        for key, value in fedgm_cfg.items():
            if not hasattr(args, key):
                setattr(args, key, value)

        current_method = getattr(args, 'method', fedgm_cfg.get('method', 'GCond'))
        
        if current_method == 'DosCond':
            print("⚠️ Detected DosCond method: Overriding config parameters (op_epoche=1, server_condense_iters=1)")
            
            setattr(args, 'op_epoche', 1)
            setattr(args, 'server_condense_iters', 1)

            fedgm_cfg['op_epoche'] = 1
            fedgm_cfg['server_condense_iters'] = 1

    except ImportError:
        print("Warning: Could not import fedgm_config")
if args.seed != 0:
    seed_everything(args.seed)

print(args)

trainer = FGLTrainer(args)
trainer.train()

