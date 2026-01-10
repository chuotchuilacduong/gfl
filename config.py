import argparse
from flcore.hyperion.hyperion_config import add_hyperion_args
# scenarios
supported_scenario = ["graph_fl", "subgraph_fl", "subgraph_fl_hetero"]

# datasets
supported_graph_fl_datasets = [
"AIDS", "BZR", "COLLAB", "COX2", "DD", "DHFR", "ENZYMES", "IMDB-BINARY", "IMDB-MULTI", "MUTAG", "NCI1", "PROTEINS", "PTC_MR", "hERG"
]
supported_subgraph_fl_datasets = [
"Cora", "CiteSeer", "PubMed", "CS", "Physics", "Computers", "Photo", "Chameleon", "Squirrel", "ogbn-arxiv", "ogbn-products", "Tolokers", "Actor", \
"Amazon-ratings", "Roman-empire", "Questions", "Minesweeper"]

supported_subgraph_fl_hetero_datasets = [
"AIFB", "BGS", "MUTAG", "ACM", "DBLP", "Freebase", "OGB-MAG", "IMDB", "DBLP_MAGNN"
]

# simulations
supported_graph_fl_simulations = ["graph_fl_cross_domain", "graph_fl_label_skew", "graph_fl_topology_skew", "graph_fl_feature_skew"]
supported_subgraph_fl_simulations = ["subgraph_fl_label_skew", "subgraph_fl_louvain_plus", "subgraph_fl_metis_plus", "subgraph_fl_louvain", "subgraph_fl_metis"]
supported_subgraph_fl_hetero_simulations = ["subgraph_fl_hetero_random_edge", "subgraph_fl_hetero_random_edge_type", "subgraph_fl_hetero_label_skew"]

# tasks
supported_graph_fl_tasks = ["graph_cls", "graph_reg"]
supported_subgraph_fl_tasks = ["node_cls", "link_pred", "node_clust"]
supported_subgraph_fl_hetero_tasks = ["node_cls_heterogeneous"]


# algorithm and models
supported_fl_algorithm = ["isolate", "fedavg", "fedprox",
                          "scaffold", "moon", "feddc", "fedproto",
                          "fedtgp", "fedpub", "fedstar", "fedgta", "fedtad",
                          "gcfl_plus", "fedsage_plus", "adafgl", "feddep", "fggp",
                          "fgssl", "fedgl", "fedhgn3", "fedgm","fedrgd", "hyperion",
                          "fedigl", "fedlog", "fedomg","fedaux","centralized"]
supported_metrics = ["accuracy", "precision", "f1", "recall", "auc", "ap", "clustering_accuracy", "nmi", "ari"]
supported_evaluation_modes = ["global_model_on_local_data", "global_model_on_global_data", "local_model_on_local_data", "local_model_on_global_data"]
supported_data_processing = ["raw", "random_feature_sparsity", "random_feature_noise", "random_topology_sparsity", "random_topology_noise", "random_label_sparsity", "random_label_noise"]

# others
supported_metapath = {'DBLP':[[('author', 'paper'), ('paper', 'author')],
                             [('author', 'paper'), ('paper', 'venue'), ('venue', 'paper'), ('paper', 'author')],
                             [('author', 'paper'), ('paper', 'term'), ('term', 'paper'), ('paper', 'author')]],
                      'ACM':[[('paper', 'author'), ('author', 'paper')],
                             [('paper', 'cite', 'paper'),('paper', 'ref', 'paper')],
                             [('paper', 'term'), ('term', 'paper')],
                             [('paper', 'subject'), ('subject', 'paper')]],
                      'Freebase':[[('book', 'book'), ('book', 'book')],
                                  [('book', 'film'), ('film', 'book')],
                                  [('book', 'business'), ('business', 'book')],
                                  [('book', 'people'), ('people', 'book')]],
                      'OGB-MAG':[[('paper', 'author'), ('author', 'paper')],
                                 [('paper', 'field_of_study'), ('field_of_study', 'paper')],
                                 [('paper', 'paper'), ('paper', 'paper')],
                                 [('paper', 'author'), ('author', 'institution'), ('institution', 'author'), ('author', 'paper')]],
                      'IMDB': [[('movie', 'actor'), ('actor', 'movie')],
                              [('movie', 'director'), ('director', 'movie')]],
                      'DBLP_MAGNN': [[('author', 'paper'), ('paper', 'author')],
                             [('author', 'paper'), ('paper', 'conference'), ('conference', 'paper'), ('paper', 'author')],
                             [('author', 'paper'), ('paper', 'term'), ('term', 'paper'), ('paper', 'author')]]}




supported_fedgraph_task = ["graph_cls", "graph_reg"]


parser = argparse.ArgumentParser()

# environment settings
parser.add_argument("--use_cuda", type=bool, default=True)
parser.add_argument("--gpuid", type=int, default=1)
parser.add_argument("--seed", type=int, default=2024)

# global dataset settings 
parser.add_argument("--root", type=str, default="/home/zhanghao/fgl/OpenFGL/openfgl/dataset")
parser.add_argument("--scenario", type=str, default="subgraph_fl", choices=supported_scenario)
parser.add_argument("--dataset", type=str, default=[], action='append')
parser.add_argument("--processing", type=str, default="raw", choices=supported_data_processing)
parser.add_argument("--processing_percentage", type=float, default=0.1)



# post_process: 
# random feature mask ratio
parser.add_argument("--feature_mask_prob", type=float, default=0.1)
# dp parameter: epsilon, support 1) random response for link
parser.add_argument("--dp_epsilon", type=float, default=0.)
# homo/hete random injection
parser.add_argument("--homo_injection_ratio", type=float, default=0.)
parser.add_argument("--hete_injection_ratio", type=float, default=0.)

# fl settings
parser.add_argument("--num_clients", type=int, default=10)
parser.add_argument("--num_rounds", type=int, default=5)
parser.add_argument("--fl_algorithm", type=str, default="fedgc", choices=supported_fl_algorithm)
parser.add_argument("--client_frac", type=float, default=1.0)


# simulation settings
parser.add_argument("--simulation_mode", type=str, default="subgraph_fl_louvain", choices=supported_graph_fl_simulations + supported_subgraph_fl_simulations)
parser.add_argument("--dirichlet_alpha", type=float, default=10)
parser.add_argument("--dirichlet_try_cnt", type=int, default=100)
parser.add_argument("--least_samples", type=int, default=5)
parser.add_argument("--louvain_resolution", type=float, default=1)
parser.add_argument("--louvain_delta", type=float, default=20, help="Maximum allowable difference in node counts between any two clients in the graph_fl_louvain simulation.")
parser.add_argument("--metis_num_coms", type=int, default=100)

# task settings
parser.add_argument("--task", type=str, default="node_cls", choices=supported_graph_fl_tasks + supported_subgraph_fl_tasks)
parser.add_argument("--num_clusters", type=int, default=7)
# training settings
parser.add_argument("--train_val_test", type=str, default="default_split")
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--optim", type=str, default="adam")
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--batch_size", type=int, default=128)


# model settings
parser.add_argument("--model", type=str, default=[], action='append')
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--hid_dim", type=int, default=256)

# evaluation settings
parser.add_argument("--metrics", type=str, default=[], action='append')
parser.add_argument("--evaluation_mode", type=str, default="local_model_on_local_data", choices=supported_evaluation_modes)

# privacy
parser.add_argument("--dp_mech", type=str, default='no_dp')
parser.add_argument("--noise_scale", type=float, default=1.0)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--dp_q", type=float, default=0.1)
# for node-level and link-level prediction tasks
parser.add_argument("--max_degree", type=int, default=5)
parser.add_argument("--max_epsilon", type=float, default=20)

# debug
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--log_root", type=str, default=None)
parser.add_argument("--log_name", type=str, default=None)
parser.add_argument("--comm_cost", type=bool, default=False)
parser.add_argument("--model_param", type=bool, default=False)

# FedRGD settings
parser.add_argument("--num_global_syn_nodes", type=int, default=100)
parser.add_argument("--server_condense_iters", type=int, default=50)
parser.add_argument("--condense_iters", type=int, default=50)
parser.add_argument("--local_epochs", type=int, default=3)
parser.add_argument("--method", type=str, default="GCond", choices=["GCond", "SGDD"], help="Method for graph condensation in FedRGD")

# IGNR / SGDD settings
parser.add_argument("--ep_ratio", type=float, default=0.5, help="Ratio for node feature in IGNR")
parser.add_argument("--sinkhorn_iter", type=int, default=5, help="Sinkhorn iterations for IGNR")
parser.add_argument("--mx_size", type=int, default=100, help="Max size for IGNR transport plan")
parser.add_argument("--opt_scale", type=float, default=1.0, help="Scaling factor for optimal transport loss")


#FedIGL settings
parser.add_argument('--lambda1', type=float, default=1.0, help='Regularization coefficient 1 for FedIGL')
parser.add_argument('--lambda2', type=float, default=1.0, help='Regularization coefficient 2 for FedIGL')
parser.add_argument('--lambda3', type=float, default=0.15, help='Weight for the regularization loss term')
parser.add_argument('--subgraph_ration', type=float, default=0.25, help='Subgraph ratio for FedIGL')

# FedLoG settings
parser.add_argument('--pre_gen_epochs', type=int, default=100, help='Epochs for generator pretraining')
parser.add_argument('--pre_epochs', type=int, default=100, help='Epochs for local model pretraining')
parser.add_argument('--head_deg_thres', type=int, default=3, help='Degree threshold for head/tail separation')
parser.add_argument('--hyper_metric', type=float, default=1, help='Hyperparameter for metric loss')
parser.add_argument('--hyper_syn_norm', type=float, default=0.5, help='Hyperparameter for synthetic norm loss')
parser.add_argument('--hyper_cd_metric', type=float, default=1, help='Hyperparameter for cross-domain metric')
parser.add_argument('--num_proto', type=int, default=5, help='Number of prototypes')

# FedOMG settings
parser.add_argument("--omg_meta_lr", type=float, default=0.1)
parser.add_argument("--grad_balance", type=bool, default=False)
parser.add_argument("--omg_learning_rate", type=float, default=0.001)
parser.add_argument("--omg_step_size", type=int, default=10)
parser.add_argument("--omg_c", type=float, default=0.1)
parser.add_argument("--omg_rounds", type=int, default=50)
parser.add_argument("--omg_momentum", type=float, default=0.9)
parser.add_argument("--omg_gamma", type=float, default=0.1)
parser.add_argument("--gm_flag", type=str, default="gem", choices=["gem", "ca_grad"])


# FedAux settings
parser.add_argument("--sigma", type=float, default=1.0, help="Sigma for FedAux kernel aggregator")
parser.add_argument("--norm_scale", type=float, default=10.0, help="Scale for exponential normalization in FedAux")
parser.add_argument("--loc_l2", type=float, default=1e-5, help="L2 regularization coefficient for FedAux")
parser.add_argument("--dropout1", type=float, default=0.5, help="Dropout rate for GNNAUX")
#Hyperion setting
parser = add_hyperion_args(parser)
args = parser.parse_args()