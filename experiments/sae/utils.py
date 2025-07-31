import pandas as pd
import numpy as np
import os
import random
import torch

from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SAE_MODEL_MAP = {
    "VanillaSAE": VanillaSAE,
    "TopKSAE": TopKSAE,
    "BatchTopKSAE": BatchTopKSAE,
    "JumpReLUSAE": JumpReLUSAE,
}