#%%
from main import MLP
import torch
import matplotlib.pyplot as plt
from putils import open_image, show_images
import os
import torch.fft as fft
import deepinv as dinv
from torchvision.utils import save_image
from deepinv.physics.generator import DiffractionBlurGenerator
from torch.optim import  LBFGS
from objectives_function import LossFidelity, blur_fn_invariant

absolute_path = os.path.abspath(os.path.dirname(__file__))
figure_path = os.path.abspath(os.path.join(absolute_path, ".."))
figure_path = os.path.join(figure_path, "tex/figures/physics_informed_invariant")

exp_type = 'simulation'
figure_path = os.path.join(figure_path, exp_type)

os.makedirs(figure_path, exist_ok=True)

input_dim = 256
output_dim = 5
model = MLP(input_dim=input_dim, output_dim=output_dim)

input_tensor = torch.randn(1, 256)
output = model(input_tensor)
print(output.shape)
# %%
