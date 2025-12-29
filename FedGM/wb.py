import wandb

ENTITY = "chuotchuilacduong-hanoi-university-of-science-and-technology"
PROJECT = "FGL-Experiment"
# ----------------------------

def rename_runs():
    api = wandb.Api()
    
    path = f"{ENTITY}/{PROJECT}"
    print(f"🔄 Đang kết nối tới: {path} ...")
    
    try:
        runs = api.runs(path)
    except Exception as e:
        print(f"❌ Lỗi kết nối: {e}")
        print("💡 Gợi ý: Kiểm tra lại tên Entity hoặc Project, và đảm bảo bạn đã 'wandb login' trên máy này.")
        return

    count = 0

    for run in runs:
        if "fedrgd" in run.name:
            old_name = run.name
            new_name = old_name.replace("fedrgd", "fedgc")
            
            run.name = new_name
            run.update() 
            
            print(f"✅ Run ID [{run.id}]: Đổi '{old_name}' -> '{new_name}'")
            count += 1
            
    if count == 0:
        print("⚠️ Không tìm thấy run nào có tên chứa 'fedrgd'.")
    else:
        print(f"\n🎉 Xong! Đã đổi tên thành công {count} runs.")

if __name__ == "__main__":
    rename_runs()