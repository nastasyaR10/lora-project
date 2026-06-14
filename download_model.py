from huggingface_hub import snapshot_download
import os
import sys

if len(sys.argv) < 3:
    print("Шаблон исполнения:")
    print("  python download_model.py <MODEL_ID> <LOCAL_DIR>")
    sys.exit(1)

MODEL_ID = sys.argv[1]
LOCAL_DIR = sys.argv[2]

print(f"Загрузка {MODEL_ID}...")
print(f"В директорию: {LOCAL_DIR}")
os.makedirs(LOCAL_DIR, exist_ok=True)

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,
    resume_download=True
)
print("Модель загружена")