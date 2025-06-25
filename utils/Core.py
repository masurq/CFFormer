import torch
import numpy as np
import random
import os


def init_random_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    torch.backends.cudnn.enabled = True


    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
