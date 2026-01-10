import subprocess
import sys
import os
# algorithms= ["fedrgd"]
# algorithms = [ "fedpub", "fedgta", "fedsage_plus", "fedavg","hyperion"]
# algorithms = ["fedgm", "fedrgd", "fedpub", "fedgta", "fedsage_plus", "fedavg"]
algorithms = ["fedgta","fedpub","fedaux","fedavg"]
client_counts = [3]
data=["Cora","CiteSeer","PubMed","Actor","Chameleon","Computers"]
# data=["Actor","Chameleon"]
# data=["Cora"]
base_args = [
    "--scenario", "subgraph_fl",
    "--simulation_mode", "subgraph_fl_label_skew",
    # "--dataset", "PubMed",          
    "--model", "gcn",              
    "--task", "node_cls",          
    "--num_epochs", "3",
    "--num_rounds", "100",
    # "--lr", "0.001",    
    "--louvain_resolution", "1",
    '--method',"GCond",
    "--seed", "2025",
    "--gpuid", "0",
    '--metrics', 'accuracy',
    "--debug", "False"
]

script_path = "main.py"

if not os.path.exists(script_path):
    print(f"Lỗi: Không tìm thấy file tại {script_path}")
    print("Vui lòng đặt script này ở thư mục cha chứa thư mục 'FedGM'.")
    sys.exit(1)

print("=== BẮT ĐẦU CHẠY ===")

for algo in algorithms:
    if algo == "fedgm":
        eval_mode = "global_model_on_local_data"        
    elif algo == "fedrgd":
        eval_mode = "local_model_on_local_data"
    else:
        eval_mode = "local_model_on_local_data"
    if algo == "fedsage_plus":
        lrate= "0.0001"
    elif algo == "fedrgd":
        lrate= "0.005"
    else: lrate = "0.001"
        
    for n_clients in client_counts:
        print(f"\n[INFO] : Algo={algo} | Clients={n_clients} | Mode={eval_mode}")
        for d in data:
            cmd = [sys.executable, script_path] + base_args + [
                "--fl_algorithm", algo,
                "--num_clients", str(n_clients),
                "--evaluation_mode", eval_mode,
                "--lr",lrate,
                "--dataset",d
            ]
            
            try:
                subprocess.run(cmd, check=True)
                print(f"[SUCCESS] {algo} với {n_clients} clients.")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR]  {algo} với {n_clients} clients.")
                continue 
print("\n=== ĐÃ HOÀN THÀNH TẤT CẢ  ===")