import os
import random
import numpy as np
import torch

def seed_everything(seed: int=42) -> None:
    '''
    Fix every random seed to unsure the reproducibility.
    Usage: Run this function in early stage of main training script
    '''
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    
def seed_worker(worker_id):
    """
    DataLoader worker initializing function.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)