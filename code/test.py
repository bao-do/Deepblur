#%%
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

absolute_path = os.path.abspath(os.path.dirname(__file__))



# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
kwargs = {'device': device, 'dtype': dtype}
psf_size = (31,31)

img = open_image(os.path.join(absolute_path, "data/first_img.JPEG"), 
                 img_size=(256, 256),
                 **kwargs)
img_size = img.shape[-2:]   
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
# sigma_list = torch.tensor([0, 0.01, 0.05, 0.1], **kwargs)
sigma_list = torch.tensor([0.1], **kwargs)

ys = y.expand(len(sigma_list), -1, -1, -1)
ys = ys + sigma_list.view(-1, 1, 1, 1) * torch.randn_like(y)
show_images(ys, title=[rf"$\sigma={sigma.item():.2f}$" for sigma in sigma_list])





# %%
from algorithm import LbfgsPsfCalibration

psfcalib = LbfgsPsfCalibration(device=device,
                               dtype=dtype,
                               psf_size=psf_size,
                               num_coeffs=15)
coeffs_est = psfcalib.forward(img.expand(len(sigma_list), -1, -1, -1),
                              ys,
                              initialization_method='sobol',
                              niter=30)
kernels_est = psfcalib.generate_blur(coeffs_est)['filter']
# show_images([kernel, kernel_est],
#             title=["original","estimated"],
#             suptitle="relative error: {:.5f}".format(torch.norm(kernel-kernel_est)/torch.norm(kernel)))

for sigma, kernel_est in zip(sigma_list, kernels_est):
    relative_error_lbfgs = (kernel-kernel_est.unsqueeze(0)).abs().sum()
    show_images([kernel, kernel_est.unsqueeze(0)],
                title=["original","estimated"],
                suptitle=f"Sigma={sigma.item():.2f}, Relative error: {relative_error_lbfgs:.5f}")
# %%
from deepinv.physics.functional import conv2d_fft
from torch.quasirandom import SobolEngine
num_coeffs = 15

objective_fn = LossFidelity(reduction="sum",
                            norm="l2",
                            physics=conv2d_fft,
                            **kwargs)


kernel_est_generator = DiffractionBlurGenerator(psf_size=psf_size,
                                            max_zernike_amplitude=0.3,
                                            zernike_index=range(2,2+num_coeffs),
                                            num_channels=1,
                                            **kwargs)

niter = 10
learning_rate = 1e-3
eta_min = 1e-8

n_restarts = 5

best_kernel_est_list = []


for (sigma, blur_ref) in zip(sigma_list, ys):
    print(f"Optimizing for sigma={sigma.item():.2f}")


    rel_err_sig_lbfgs = []

    ################# LBFGS optimization with multiple restarts #################
    best_loss = float('inf')
    best_kernel_est = None
    
    # initialize inital points covering the search space, using sobol sequence

    sobol_engine = SobolEngine(dimension=num_coeffs,
                            scramble=True,
                            seed=random_seed())
    for restart in range(n_restarts):
        print(f"Restart {restart+1}/{n_restarts}")
        coeffs = sobol_engine.draw(1).to(**kwargs)
        coeffs = max_zernike_amplitude * (coeffs - 0.5)
        coeffs = coeffs.requires_grad_(True)
        optimizer = LBFGS([coeffs],
                        lr=1.0,
                        history_size=10,
                        max_iter=20,
                        line_search_fn="strong_wolfe")
        loss_iter = []
        for i in range(niter):
            
            def closure():
                filters = kernel_est_generator.step(batch_size=1,
                                                coeff=coeffs)['filter']
                optimizer.zero_grad()
                loss = objective_fn(img, blur_ref.unsqueeze(0), filter=filters, crop=False)
                loss.backward()
                return loss
            optimizer.step(closure)
            loss = closure()
            loss_iter.append(loss.item())
            if (i % 10 == 0) or (i == niter - 1):
                print(f"Iteration {i+1}/{niter}, Loss: {loss_iter[-1]:.4f}")

        

        if loss_iter[-1] < best_loss:
            best_loss = loss_iter[-1]
            best_loss_iter = loss_iter
            best_kernel_est = kernel_est_generator.step(batch_size=1,
                                                    coeff=coeffs)['filter'].detach()

    relative_error_lbfgs = (kernel-best_kernel_est).abs().sum()

    show_images([kernel, best_kernel_est],
                title=["original","estimated"],
                suptitle="relative error: {:.5f}".format(relative_error_lbfgs))
# %%
