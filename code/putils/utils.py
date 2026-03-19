import torch
def random_seed():
    return torch.randint(0, 10000, (1,)).item()