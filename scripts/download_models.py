from huggingface_hub import snapshot_download

from mini_infer.loader import MODEL_FILES

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]

for repo in MODELS:
    path = snapshot_download(repo, allow_patterns=MODEL_FILES)
    print(repo, "->", path)
