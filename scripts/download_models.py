from huggingface_hub import snapshot_download

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]

for repo in MODELS:
    path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    print(repo, "->", path)
