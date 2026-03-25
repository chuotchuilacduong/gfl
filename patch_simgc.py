import os

root_dir = r"g:\Code\python\FedGM\flcore\fedgvd"
client_file = os.path.join(root_dir, "client.py")
simgc_file = os.path.join(root_dir, "SimGC_transductive.py")

# PATCH CLIENT.PY path formatting
with open(client_file, "r", encoding="utf-8") as f:
    client_content = f.read()
client_content = client_content.replace("{args.dataset}", "{args.dataset[0]}")
with open(client_file, "w", encoding="utf-8") as f:
    f.write(client_content)


# PATCH SIMGC_transductive.py
with open(simgc_file, "r", encoding="utf-8") as f:
    simgc_content = f.read()

# Fix loop overrides
simgc_content = simgc_content.replace("args.teacher_model_loop = getattr(args, 'num_epochs', 2)", "args.teacher_model_loop = 1000")
simgc_content = simgc_content.replace("args.condensing_loop = getattr(args, 'num_epochs', 2)", "args.condensing_loop = 150") # Set to 150 for testing speed
simgc_content = simgc_content.replace("args.student_model_loop = getattr(args, 'num_epochs', 2)", "args.student_model_loop = 300")

# Fix missing underscores
simgc_content = simgc_content.replace("str(args.seed)+str(client_id)+'.pt'", "str(args.seed)+'_'+str(client_id)+'.pt'")

# Fix array string formatting
simgc_content = simgc_content.replace("{args.dataset}_", "{args.dataset_str}_")

# Add folder creation
dir_creation = """
import os
os.makedirs(os.path.join(root, 'saved_ours'), exist_ok=True)
os.makedirs(os.path.join(root, 'saved_model', 'teacher'), exist_ok=True)
os.makedirs(os.path.join(root, 'saved_model', 'student'), exist_ok=True)
"""
if "os.makedirs(os.path.join(root, 'saved_ours')" not in simgc_content:
    simgc_content = simgc_content.replace("from flcore.fedgvd.models.parametrized_adj import PGE", "from flcore.fedgvd.models.parametrized_adj import PGE\n" + dir_creation)

with open(simgc_file, "w", encoding="utf-8") as f:
    f.write(simgc_content)

print("Patching complete.")
