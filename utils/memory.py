import gc
import torch

def free_memory():
    """Free up GPU memory aggressively."""
    gc.collect()
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        # Force a sync point
        torch.cuda.synchronize()