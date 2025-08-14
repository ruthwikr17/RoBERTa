import gc
import torch

gc.collect()
torch.mps.empty_cache()
print("Memory cleared!")
