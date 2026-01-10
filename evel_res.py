import wandb
import pandas as pd
import numpy as np

# --- CẤU HÌNH (Lấy từ URL trong ảnh của bạn) ---
# URL: wandb.ai/chuotchuilacduong-hanoi-university-of-science-and-technology/FGL-Experiment
ENTITY = "chuotchuilacduong-hanoi-university-of-science-and-technology"
PROJECT = "FGL-Experiment"

METRIC_NAME = "current_test_accuracy"
TARGET_CLIENTS = [5, 10, 20]  # Số lượng client cần lấy

# Map tên thuật toán từ code (WandB config) sang tên hiển thị trong báo cáo
ALGO_MAPPING = {
    "fedavg": "FedAvg",
    "fedsage_plus": "FedSage+",
    "fedpub": "FedPub",
    "fedgm": "FedGM",
    "fedgta": "FedGTA",
    "fedomg": "FedOMG",
    "fedaux": "FedAux",
    "fedigl": "FedIGL",
    "hyperion": "Hyperion",
    "fedlog": "FedLoG",
    "fedrgd": "FedGC", # Theo yêu cầu của bạn
    # Thêm các tên khác nếu cần
}

# Thứ tự dòng mong muốn (như trong ảnh mẫu)
ROW_ORDER = [
    "Central GNN", "FedAvg", "FedSage+", "FedPub", "FedGM", 
    "FedGTA", "FedOMG", "FedAux", "FedIGL", "Hyperion", 
    "FedLoG", "FedGC", "Local GNN"
]

def get_wandb_data():
    api = wandb.Api()
    # Lấy toàn bộ runs trong project
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    
    data = []
    print(f"Đang tải dữ liệu từ {len(runs)} runs...")
    
    for run in runs:
        # Lấy config
        cfg = run.config
        
        # 1. Xác định Algorithm (ưu tiên map sang tên đẹp)
        # Kiểm tra nhiều key phổ biến: 'algorithm', 'method', 'model'
        raw_algo = cfg.get('algorithm') or cfg.get('method') or "Unknown"
        display_algo = ALGO_MAPPING.get(raw_algo, raw_algo) # Nếu không map thì giữ nguyên
        
        # 2. Xác định Dataset và Num Clients
        dataset = cfg.get('dataset', 'Unknown')
        n_clients = cfg.get('num_clients')
        
        # Chỉ lấy đúng các mốc client 5, 10, 20
        if n_clients not in TARGET_CLIENTS:
            continue
            
        # 3. Lấy Metric
        if METRIC_NAME in run.summary:
            val = run.summary[METRIC_NAME]
            # Chuyển về % nếu đang ở dạng 0.xx
            if val <= 1.0: val *= 100
            
            data.append({
                "Algorithm": display_algo,
                "Dataset": dataset,
                "Clients": n_clients,
                "Score": val
            })
            
    return pd.DataFrame(data)

def create_report_table(df):
    if df.empty: return pd.DataFrame()

    # Tính Mean và Std
    grouped = df.groupby(['Algorithm', 'Dataset', 'Clients'])['Score'].agg(['mean', 'std'])
    
    # Format thành chuỗi "Mean ± Std"
    # Dùng fillna(0) cho std nếu chỉ chạy 1 seed
    grouped['result_str'] = grouped.apply(
        lambda x: f"{x['mean']:.2f} $\pm$ {x['std']:.2f}" if not pd.isna(x['std']) else f"{x['mean']:.2f}", 
        axis=1
    )
    
    # Pivot table: Dòng=Algo, Cột=[Dataset, Clients]
    pivot_df = grouped.reset_index().pivot(
        index='Algorithm',
        columns=['Dataset', 'Clients'],
        values='result_str'
    )
    
    # Sắp xếp lại thứ tự dòng theo ROW_ORDER
    # Chỉ giữ lại các thuật toán có trong dữ liệu
    existing_algos = [algo for algo in ROW_ORDER if algo in pivot_df.index]
    # Thêm các thuật toán lạ (nếu có) vào cuối
    other_algos = [algo for algo in pivot_df.index if algo not in ROW_ORDER]
    
    pivot_df = pivot_df.reindex(existing_algos + other_algos)
    
    # Điền dấu "-" cho ô trống
    pivot_df = pivot_df.fillna("-")
    
    return pivot_df

# --- CHẠY CHƯƠNG TRÌNH ---
df_raw = get_wandb_data()

if not df_raw.empty:
    final_table = create_report_table(df_raw)
    
    print("\n--- BẢNG TỔNG HỢP (Copy vào báo cáo) ---")
    print(final_table)
    
    # Xuất ra CSV
    final_table.to_csv("fgl_results_matrix.csv")
    print("\nĐã lưu file 'fgl_results_matrix.csv'")
    
    # Xuất ra LaTeX code
    print("\n--- LATEX CODE ---")
    print(final_table.to_latex())
else:
    print("Không tìm thấy dữ liệu. Kiểm tra lại tên Entity/Project trong code.")