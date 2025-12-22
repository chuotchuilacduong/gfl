import subprocess
import sys
import os

# 1. Cấu hình các thuật toán và tham số
# Lưu ý: Tên thuật toán phải khớp với 'supported_fl_algorithm' trong config.py
algorithms = ["fedgm", "fedrgd", "fedpub", "fedgta", "fedsage_plus", "fedavg"]
client_counts = [5, 10, 20]

# 2. Các tham số nền (giữ nguyên như trong main.py cũ của bạn hoặc tuỳ chỉnh)
# Bạn có thể thay đổi dataset, model, rounds tại đây
base_args = [
    "--scenario", "subgraph_fl",
    "--simulation_mode", "subgraph_fl_label_skew",
    "--dataset", "Computers",          # Dataset mặc định
    "--model", "gcn",              # Model mặc định
    "--task", "node_cls",          # Task mặc định
    "--num_epochs", "3",
    "--num_rounds", "100",
    "--lr", "0.001",
    "--louvain_resolution", "1",
    "--seed", "2024",
    "--gpuid", "0",
    '--metrics', 'accuracy',
    "--debug", "False"
]

# Đường dẫn đến file main.py
script_path = "main.py"

# Kiểm tra file tồn tại
if not os.path.exists(script_path):
    print(f"Lỗi: Không tìm thấy file tại {script_path}")
    print("Vui lòng đặt script này ở thư mục cha chứa thư mục 'FedGM'.")
    sys.exit(1)

print("=== BẮT ĐẦU CHẠY THỰC NGHIỆM ===")

for algo in algorithms:
    # 3. Xử lý ràng buộc về evaluation_mode
    if algo == "fedgm":
        eval_mode = "global_model_on_local_data"
    else:
        # Các thuật toán khác dùng local_model_on_local_data như yêu cầu
        eval_mode = "local_model_on_local_data"

    for n_clients in client_counts:
        print(f"\n[INFO] Đang chạy: Algo={algo} | Clients={n_clients} | Mode={eval_mode}")
        
        # 4. Xây dựng câu lệnh
        cmd = [sys.executable, script_path] + base_args + [
            "--fl_algorithm", algo,
            "--num_clients", str(n_clients),
            "--evaluation_mode", eval_mode
        ]
        
        # 5. Thực thi
        try:
            # check=True sẽ báo lỗi nếu script con bị crash
            subprocess.run(cmd, check=True)
            print(f"[SUCCESS] Hoàn thành {algo} với {n_clients} clients.")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Lỗi khi chạy {algo} với {n_clients} clients.")
            # Có thể dùng 'continue' để bỏ qua lỗi và chạy tiếp cái sau, 
            # hoặc 'break' để dừng lại kiểm tra.
            continue 

print("\n=== ĐÃ HOÀN THÀNH TẤT CẢ CÁC THỰC NGHIỆM ===")