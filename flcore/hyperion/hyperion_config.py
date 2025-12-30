import argparse

def add_hyperion_args(parser):
    """
    Adds Hyperion-specific arguments to the global parser.
    """
    group = parser.add_argument_group('Hyperion Algorithm Parameters')

    # Model arguments
    group.add_argument('--num_prototypes_per_class', type=int, default=5, 
                       help='Number of prototypes per class.')
    group.add_argument('--prot_dim', type=int, default=64, 
                       help='Dimension of the prototype vector.')
    
    # Loss weights & Hyperparameters
    group.add_argument('--clst', type=float, default=0.1, 
                       help='Weight for clustering loss.')
    group.add_argument('--sep', type=float, default=0.1, 
                       help='Weight for separation loss.')
    group.add_argument('--proto_contrast_weight', type=float, default=0.1, 
                       help='Weight for prototype contrastive loss.')
    group.add_argument('--alpha', type=float, default=0.5, 
                       help='Weight for dynamic cross-entropy loss.')
    group.add_argument('--tau', type=float, default=0.5, 
                       help='Temperature parameter for contrastive loss.')
    group.add_argument('--p', type=float, default=0.2, 
                       help='Dropout probability for data augmentation (edge/feature).')

    # Knowledge Distillation (WKD) & Sinkhorn parameters
    group.add_argument('--wkd_temperature', type=float, default=1.0, 
                       help='Temperature for WKD (Weighted Knowledge Distillation).')
    group.add_argument('--sinkhorn_lambda', type=float, default=0.05, 
                       help='Lambda regularization for Sinkhorn algorithm.')
    group.add_argument('--wkd_sinkhorn_iter', type=int, default=10, 
                       help='Maximum iterations for Sinkhorn algorithm.')
    group.add_argument('--wkd_logit_weight', type=float, default=1.0, 
                       help='Weight for WKD logit loss.')
    
    #Scheduler for Distillation
    group.add_argument('--wkd_loss_cosine_decay_epoch', type=int, default=50, 
                       help='Epoch to start cosine decay for WKD loss weight.')
    group.add_argument('--solver_epochs', type=int, default=100, 
                       help='Total epochs used for calculating cosine decay schedule.')
    
    
    # Malicious Detection & Pruning
    group.add_argument('--pruning_epochs', type=int, default=10, 
                       help='Number of rounds before starting the pruning process.')
    group.add_argument('--retain_ratio', type=float, default=0.8, 
                       help='Ratio of nodes to retain during pruning for potential poisoned clients.')

    return parser