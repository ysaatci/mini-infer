import torch

assert torch.cuda.is_available(), "CUDA not available"
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("matmul ok", (x @ x).float().abs().mean().item())
