config = {
    # Chọn phương pháp: 'GCond', 'SGDD', 'FedGM', hoặc phương pháp mới của bạn
    "method": "GCond", 
    
    "reduction_rate": 0.84,
    "op_epoche": 100,
    "opti_pge_epoche": 5,
    "opti_x_epoche": 15,
    "lr_feat": 1e-2,
    "lr_adj": 1e-2,
    "one_step": True,
    "alpha": 0, 
    "batch_real": 256,
    "dis_metric": "mse",
    
    "ep_ratio": 0.5,      
    "sinkhorn_iter": 5,    
    "mx_size": 100,      
    "opt_scale": 1.0,     
}