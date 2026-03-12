#%%
import os
project_abs_path = os.path.abspath(os.path.dirname(__file__)) 
import torch
from putils import show_images, open_image
from deepinv.physics import TiledSpaceVaryingBlur
from deepinv.physics.generator import DiffractionBlurGenerator
from torchmetrics.image import TotalVariation, PeakSignalNoiseRatio
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from objectives_function import LossFidelity, RegImage, RegFilter, TotalLoss, grad



# %% Load image
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
reduction = 'sum'
factorial_kwargs = {'device': device,
                    'dtype': dtype,
                    'reduction': reduction}



kwargs = {'device': device, 'dtype': dtype}

patch_size = (64, 64)
stride = (32, 32)

kernel_size = (31,31)
num_kernels_x = 15
num_kernels_y = 15
num_kernels = num_kernels_x * num_kernels_y 
channels = 1
pupil_size = (256, 256)

abbrev_image = open_image(os.path.join(project_abs_path, "data/80-astig.png"),
                        **kwargs)
img_size = abbrev_image.shape[-2:]

nonabbrev_image = open_image(os.path.join(project_abs_path, "data/0-astig.png"),
                        **kwargs)

show_images([abbrev_image, nonabbrev_image],
            title=["Abbreviated Image", "Non-abbreviated Image"])
# %% Generate diffraction blur kernels
generator = DiffractionBlurGenerator(kernel_size, channels, pupil_size=pupil_size, **kwargs)
print("\n".join(generator.zernike_polynomials)) # list of Zernike polynomials used


#%% Blind deblurring 
# initialization
basis = "zernike"
generator = DiffractionBlurGenerator(kernel_size,
                                     channels,
                                     pupil_size=pupil_size,
                                     **kwargs)
coeffs = generator.generate_coeff(batch_size=num_kernels)*100
coeffs = coeffs.requires_grad_(True)

x = torch.clone(nonabbrev_image.detach()).requires_grad_(True)

physics = TiledSpaceVaryingBlur(patch_size=patch_size, stride=stride, **kwargs)

loss_fidel = LossFidelity(physics=physics, **factorial_kwargs)

filters_reg = RegFilter(kernel_size=kernel_size,
                        num_kernels=(num_kernels_x, num_kernels_y),
                        reg_coeffs=(1.0, 1.0, 1.0),
                        r=3,
                        **factorial_kwargs)

img_reg = RegImage(**factorial_kwargs)

lamb = 1
# lamb_min = 1
gamma = 0.1

K = 500
K_x = 10
n_reset = 100
step_kernel = 0.01

optimizer_kernel = torch.optim.AdamW([coeffs], lr=step_kernel)
# %%

for k in range(K):
    if k % n_reset == 0:
        # print(f"Iteration {k}/{K}")
        # show_images([x, nonabbrev_image],
        #     title=["Deblurred Image", "Non-abbreviated Image"])
        # est_filters = generator.step(batch_size=num_kernels, coeff=coeffs)['filter']
        # # random 10 indices
        # indices = torch.randperm(num_kernels)[:10]
        # show_images(est_filters[indices,...],
        #             suptitle="Estimated Filters")
        
        x = torch.clone(abbrev_image.detach()).requires_grad_(True)

    filters_detached = generator.step(batch_size=num_kernels,
                                      coeff=coeffs.detach())['filter'].detach()
    optimizer_x = torch.optim.AdamW([x], lr=1e-2)
    for i in range(K_x):
        loss_x_fidel = loss_fidel.forward(x, abbrev_image, filters=filters_detached)
        loss_x_reg = img_reg.forward(x)
        loss_x = loss_x_fidel + lamb * loss_x_reg
    
        optimizer_x.zero_grad()
        loss_x.backward()
        optimizer_x.step()
    show_images([x, nonabbrev_image],
        title=["estimated image", "Non-abbreviated Image"])
    
    # Recompute fresh filters (attached to coeffs) for the kernel update step
    filters = generator.step(batch_size=num_kernels,
                             coeff=coeffs)['filter']
    loss_kernel_fidel = loss_fidel.forward(x, abbrev_image, filters=filters)
    loss_kernel_reg = filters_reg.forward(filters)
    loss_kernel = loss_kernel_fidel + gamma * loss_kernel_reg
    # loss_kernel = gamma * loss_kernel_reg
    
    optimizer_kernel.zero_grad()
    loss_kernel.backward()
    optimizer_kernel.step()




# %%
show_images([x, nonabbrev_image],
            title=["Deblurred Image", "Non-abbreviated Image"])

# show the gt and estimated pupil functions
est_filters = generator.step(batch_size=num_kernels, coeff=coeffs)['filter']
# random 10 indices
# indices = torch.randperm(num_kernels)[:10]
show_images(est_filters, ncols=15,
            suptitle="Estimated Filters")

# %%
