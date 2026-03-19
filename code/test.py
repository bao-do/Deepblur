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
# sigma_list = torch.tensor([0, 0.01, 0.05, 0.1], **kwargs)
sigma_list = torch.tensor([0.05], **kwargs)

ys = y.expand(len(sigma_list), -1, -1, -1)
ys = ys + sigma_list.view(-1, 1, 1, 1) * torch.randn_like(y)
show_images(ys, title=[rf"$\sigma={sigma.item():.2f}$" for sigma in sigma_list])





# %%


#%%
psfcalib = PsfCalibration(num_coeffs=30, **kwargs)
initial_coeffs = torch.randn(1, psfcalib.input_dim, **kwargs)
for sigma, blur_true in zip(sigma_list, ys):
    print(f"Optimizing for sigma={sigma.item():.2f}")
    blur_est, loss_iter = psfcalib._forward_one_image(img,
                                                    blur_true.unsqueeze(0),
                                                    niter=300,
                                                    initial_coeffs=initial_coeffs,
                                                    crop=False)
    show_images([kernel, blur_est['filter']],
                title=["Original Kernel", "Estimated Kernel"],
                suptitle=f'relative error: {torch.norm(kernel-blur_est["filter"])/torch.norm(kernel):.4f}')
    fig = plt.figure(figsize=(10, 5))
    plt.plot(loss_iter)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")  
    plt.title(f"Loss Curve")
# %%
from deepinv.physics.functional import conv2d_fft
niter=30
objective_fn = LossFidelity(reduction="sum",
                            norm="l2",
                            physics=conv2d_fft,
                            **kwargs)
kernel_est_generator = psfcalib.kernel_generator
for sigma, blur_true in zip(sigma_list, ys):
    print(f"Optimizing for sigma={sigma.item():.2f}")


    coeffs = blur_est['coeff'].detach().clone()
    coeffs = coeffs.requires_grad_(True)

    optimizer = LBFGS([coeffs],
                    lr=1,
                    history_size=10,
                    max_iter=20,
                    line_search_fn="strong_wolfe")
    loss_iter = []
    for i in range(niter):
        
        def closure():
            filters = kernel_est_generator.step(batch_size=1,
                                            coeff=coeffs)['filter']
            optimizer.zero_grad()
            loss = objective_fn(img, blur_true.unsqueeze(0), filter=filters, crop=False)
            loss.backward()
            return loss
        optimizer.step(closure)
        loss = closure()
        loss_iter.append(loss.item())
        # if (i % 10 == 0) or (i == niter - 1):
        #     print(f"Iteration {i+1}/{niter}, Loss: {loss.item():.10f}")

show_images([kernel, kernel_est_generator.step(batch_size=1, coeff=coeffs)['filter']],
            title=["Original Kernel", "LBFGS Estimated Kernel"],
            suptitle=f'relative error: {torch.norm(kernel-kernel_est_generator.step(batch_size=1, coeff=coeffs)["filter"])/torch.norm(kernel):.4f}')



# %%
