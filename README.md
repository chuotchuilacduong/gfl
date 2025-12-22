# Rethinking Federated Graph Learning: A Data Condensation Perspective
Official code repository of the paper "Rethinking Federated Graph Learning: A Data Condensation Perspective" in the proceedings of International Joint Conference on Artificial Intelligence (IJCAI) 2025.

https://arxiv.org/abs/2505.02573

## Requirement
Please see requirements.txt.

```
torch==2.0.1+cu117
torch-geometric==2.6.1
torch-scatter==2.1.2+pt20cu117
torch-sparse==0.6.18+pt20cu117
scipy==1.14.0
numpy==1.26.4
ogb==1.3.6
```
## Run the code
Here we take FedGM-Cora-Louvain-10 Clients as an example:

```
python main.py --fl_algorithm fedgm --dataset Cora --num_clients 10 --simulation_mode subgraph_fl_louvain
```

## Citation 
If you found the provided code with our paper useful in your work, we kindly request that you cite our work.

```
@misc{zhang2025rethinkingfederatedgraphlearning,
      title={Rethinking Federated Graph Learning: A Data Condensation Perspective}, 
      author={Hao Zhang and Xunkai Li and Yinlin Zhu and Lianglin Hu},
      year={2025},
      eprint={2505.02573},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.02573}, 
}
```
