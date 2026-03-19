#%%
import torch
import torch
import matplotlib.pyplot as plt
from putils import open_image, show_images
import os
import torch.fft as fft
import deepinv as dinv
from torchvision.utils import save_image
from deepinv.physics.generator import DiffractionBlurGenerator
from torch.optim import  LBFGS, AdamW
from objectives_function import LossFidelity, blur_fn_invariant
from tqdm import tqdm
import copy
import torch.nn as nn
from typing import Tuple
from neural_network import PsfCalibration
from deepinv.physics.functional import conv2d_fft

absolute_path = os.path.abspath(os.path.dirname(__file__))



# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
kwargs = {'device': device, 'dtype': dtype}
img_size = (256, 256)
psf_size = (31, 31)

img = open_image(os.path.join(absolute_path, "data/first_img.JPEG"), 
                 img_size=img_size,
                 **kwargs)
img_crop = img[..., psf_size[0]//2:-(psf_size[0]//2), psf_size[1]//2:-(psf_size[1]//2)]
show_images([img], title=["Original Image"])
# %%
max_zernike_amplitude = 0.3

filter_generator = DiffractionBlurGenerator(psf_size=psf_size,
                                            max_zernike_amplitude=max_zernike_amplitude,
                                            zernike_index=range(2, 12),
                                            num_channels=1,
                                            **kwargs)
def random_seed():
    return torch.randint(0, 10000, (1,)).item()

blur = filter_generator.step(batch_size=1, seed=random_seed())
kernel = blur['filter']
pupil = blur['pupil']
y = blur_fn_invariant(img, kernel)

show_images([torch.real(pupil)], title=["Pupil Function"])
show_images([kernel], title=["Original Kernel"])
show_images([y], title=["Blurred Image"])

# %%
sigma_list = torch.tensor([0, 0.01, 0.05, 0.1], **kwargs)
# sigma_list = torch.tensor([0.05], **kwargs)

ys = y.expand(len(sigma_list), -1, -1, -1)
ys = ys + sigma_list.view(-1, 1, 1, 1) * torch.randn_like(y)
show_images(ys, title=[rf"$\sigma={sigma.item():.2f}$" for sigma in sigma_list])

#%%
num_coeffs = 15


# %% 
import torch
from evotorch import Problem
from evotorch.algorithms import CMAES

# Define your objective function using PyTorch operations
num_coeffs = 15

filter_est_generator = DiffractionBlurGenerator(psf_size=psf_size,
                                            max_zernike_amplitude=0.3,
                                            zernike_index=range(2, 2+num_coeffs),
                                            num_channels=1,
                                            **kwargs)

objective_func = LossFidelity(reduction="sum",
                            norm="l2",
                            physics=conv2d_fft,
                            **kwargs)

for sigma, blur_ref in zip(sigma_list, ys):
    print(f"Optimizing for sigma={sigma.item():.2f}")

    def objective_func_wrapper(coeffs):
        filters = filter_est_generator.step(coeff=coeffs.unsqueeze(0))['filter']

        
        loss = objective_func(x=img,
                              y=blur_ref.unsqueeze(0),
                              filter=filters,crop=False).item() 
        return loss

    # Configure the problem domain
    problem = Problem(
        "min", 
        objective_func=objective_func_wrapper, 
        solution_length=num_coeffs, 
        initial_bounds=(-0.15, 0.15),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    searcher = CMAES(problem, stdev_init=0.001)

    searcher.run(num_generations=200)

    best_weights = searcher.status["pop_best"].values.clone()

    kernel_est = filter_est_generator.step(coeff=best_weights.unsqueeze(0))['filter']
    show_images([kernel_est, kernel], title=["Estimated Kernel", "Original Kernel"],
                suptitle=f"Relative Error: {torch.norm(kernel_est-kernel)/torch.norm(kernel):.4f}")

# %%
