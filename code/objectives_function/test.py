#%%
import os
project_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
print(project_abs_path)
import sys
sys.path.append(project_abs_path)
# %%
from PIL import Image
import torch
import torch.nn.functional as F
import deepinv as dinv
from putils import show_images
import numpy as np
from deepinv.physics import TiledSpaceVaryingBlur
from deepinv.physics.generator import DiffractionBlurGenerator
from torchmetrics.image import TotalVariation, PeakSignalNoiseRatio
from torch.optim.lr_scheduler import CosineAnnealingLR

# %% Load image
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

kwargs = {'device': device, 'dtype': dtype}

img_size = (256, 256)
patch_size = (64, 64)
stride = (32, 32)

kernel_size = (31,31)
num_kernels_x = 7
num_kernels_y = 7
num_kernels = num_kernels_x * num_kernels_y 
channels = 1
pupil_size = (256, 256)

img = Image.open(os.path.join(project_abs_path, 'data/first_img.JPEG')).convert('L')
# resize to 256x256
img = img.resize(img_size) 
# convert to torch tensor
img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(**kwargs)
show_images(img_tensor)
# %% Generate diffraction blur kernels
generator = DiffractionBlurGenerator(kernel_size, channels, pupil_size=pupil_size, **kwargs)
print("\n".join(generator.zernike_polynomials)) # list of Zernike polynomials used
#%%
blur_gt = generator.step(batch_size=num_kernels)  # dict_keys(['filter', 'coeff', 'pupil'])
filters_gt = blur_gt['filter'].transpose(0, 1).unsqueeze(0)
# %%
sigma = 0.01
physics = TiledSpaceVaryingBlur(patch_size=patch_size, stride=stride, **kwargs)
y = physics(img_tensor, filters=filters_gt)

y += sigma * torch.randn_like(y)
print(img_tensor.shape, y.shape)
dinv.utils.plot([img_tensor, y], titles=["Original", "Blurred"])

# %%
coeffs_gt = blur_gt['coeff']
print(coeffs_gt.shape)
# %%
# initialize coeffs with random values
generator = DiffractionBlurGenerator(kernel_size,
                                     channels,
                                     pupil_size=pupil_size,
                                     zernike_index=tuple(range(3,14)),
                                     **kwargs)
coeffs = generator.generate_coeff(batch_size=num_kernels).requires_grad_(True)
# coeffs = torch.randn_like(coeffs_gt).requires_grad_(True)
# compute the corresponding filters
blur = generator.step(batch_size=num_kernels, coeff=coeffs)
filters = blur['filter'].transpose(0, 1).unsqueeze(0)

x = physics(img_tensor, filters=filters)
dinv.utils.plot([img_tensor, x, y], titles=["Original", "Blurred_init", "Blurred_gt"])

alpha = 0.5
lamb = 0.1
# %%


criterion = torch.nn.MSELoss(reduction='sum')

def grad(input, dim=3):

    if isinstance(dim, (list, tuple)):
        return [grad(input, d) for d in dim]

    gindex = [slice(None)] * input.dim()
    gindex[dim] = slice(1, None)

    lindex = [slice(None)] * input.dim()
    lindex[dim] = slice(None, -1)
    return input[tuple(gindex)] - input[tuple(lindex)]

# %%
# update the coefficients using gradient descent
learning_rate = 0.01
optimizer = torch.optim.AdamW([coeffs], lr=learning_rate)
#%%
n_iter = 500
losses = []
for i in range(n_iter):
    blur = generator.step(batch_size=num_kernels, coeff=coeffs)
    filters = blur['filter'].transpose(0, 1).unsqueeze(0)
    x = physics(img_tensor, filters=filters)

    # compute fidelity loss
    loss_fidel = criterion(x, y)

    # compute regularization term 
    grad_h, grad_v = grad(filters.view(num_kernels_y, num_kernels_x, *filters.shape[-2:]), dim=(0,1))
    R1 = grad_h.pow(2).sum() + grad_v.pow(2).sum()
    R2 = torch.sum((torch.sum(filters, dim=(0,1,3,4)) - 1).pow(2))
    loss_reg = alpha * R1 + (1 - alpha) * R2

    # total loss
    L = loss_fidel + lamb * loss_reg
    losses.append(L.item())
    
    optimizer.zero_grad()
    L.backward()
    optimizer.step()

    if i % 10 == 0 or i == n_iter - 1:
        print(f"iter {i:03d}: loss_fidel={loss_fidel.item():.6f}, loss_reg={loss_reg.item():.6f}, total={L.item():.6f}")
        x_deconv = physics.A_adjoint(y, filters=filters)
        show_images([img_tensor, x_deconv], title=["Original", "estimated"])
#%%
import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()


# %%
estimated_blur = generator.step(batch_size=num_kernels, coeff=coeffs)
estimated_filters = estimated_blur['filter'].transpose(0, 1).unsqueeze(0)
show_images([physics(img_tensor, filters=estimated_filters), y], title=["Blurred with estimated filters", "Blurred with GT filters"])
# %%
img_res = physics.A_adjoint(y, filters=filters_gt)
show_images([img_tensor, img_res], title=["Original", "Deconvolved with GT"])
# %%
# show the gt and estimated pupil functions
pupil_gt = blur_gt['pupil'].unsqueeze(1)  
pupil_est = estimated_blur['pupil'].unsqueeze(1)
for i in range(10):
    show_images([torch.real(pupil_gt[i:i+1]), torch.real(pupil_est[i:i+1])], title=[f"GT Pupil {i}", f"Estimated Pupil {i}"])
    show_images([filters_gt[0, 0, i:i+1].unsqueeze(0), 
                 estimated_filters[0, 0, i:i+1].unsqueeze(0)],
                 title=[f"GT Filter {i}", f"Estimated Filter {i}"])
# %%
# Optimization algo for recover the image with the estimated filters
# physics.A_adjoint does not work well
initial_lr = 1e-2
min_lr = 1e-4
n_iter_u = 2000
# u = physics.A_adjoint(y, estimated_filters).detach().requires_grad_(True)
padding = (15,15,15,15)
u = F.pad(y, padding).detach().requires_grad_(True)
optimizer_u = torch.optim.AdamW([u], lr=initial_lr)
lr_scheduler = CosineAnnealingLR(optimizer_u, eta_min=min_lr, T_max=n_iter_u)

estimated_filters = estimated_blur['filter'].transpose(0, 1).unsqueeze(0).detach()
metric = TotalVariation().to(**kwargs)
losses_u = []
for i in range(n_iter_u):
    u_blur = physics(u, filters=estimated_filters)
    loss_u = criterion(u_blur, y) + 0.001 * metric(u)
    # loss_u = criterion(u_blur, y)
    optimizer_u.zero_grad()
    loss_u.backward()
    optimizer_u.step()
    lr_scheduler.step()
    losses_u.append(loss_u.item())
    if i % 200 == 0 or i == n_iter_u - 1:
        print(f"iter {i:03d}: loss_u={loss_u.item():.6f}")
        show_images([img_tensor, u], title=["Original", "Recovered with estimated filters"]) 

# %%
# crop the center of the estimated image and the original image for better visualization
fig = plt.figure()
plt.plot(losses_u)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Loss curve for image recovery with estimated filters")
plt.show()

psnr = PeakSignalNoiseRatio(data_range=1.0).to(**kwargs)

crop_size = y.shape[-2:]
crop_u = u[..., (u.shape[-2] - crop_size[0]) // 2 : (u.shape[-2] + crop_size[0]) // 2, (u.shape[-1] - crop_size[1]) // 2 : (u.shape[-1] + crop_size[1]) // 2]
img_tensor_crop = img_tensor[..., (img_tensor.shape[-2] - crop_size[0]) // 2 : (img_tensor.shape[-2] + crop_size[0]) // 2, (img_tensor.shape[-1] - crop_size[1]) // 2 : (img_tensor.shape[-1] + crop_size[1]) // 2]
show_images([img_tensor_crop, crop_u, y],
            title=["Original", "Recovered", "Noisy"],
            suptitle=f'psnr={psnr(crop_u, img_tensor_crop).item():.2f}dB')



# %%
# approximate the pupil function with fourier atoms

# frequency grid for a ball of radius r in the Fourier domain
n = 15

freqs_x = torch.fft.fftfreq(img_tensor.shape[-2]).to(**kwargs)
freq_x = torch.cat([freqs_x[0:n+1], freqs_x[-n:]])

freq_y = torch.fft.fftfreq(img_tensor.shape[-1]).to(**kwargs)
freq_y = torch.cat([freq_y[0:n+1], freq_y[-n:]])

freq_grid = torch.meshgrid(freq_x, freq_y)

coeffs_fourier = torch.randn((num_kernels, 2*n+1, 2*n+1),
                             dtype= torch.complex64,
                             device=device).requires_grad_(True)

# %%
x = physics(img_tensor, filters=filters)
dinv.utils.plot([img_tensor, x, y], titles=["Original", "Blurred_init", "Blurred_gt"])

#%%
n_iter = 10000
losses = []

learning_rate = 0.01
optimizer = torch.optim.AdamW([coeffs_fourier], lr=learning_rate)

for i in range(n_iter):
    rooted_filters = torch.fft.ifft2(coeffs_fourier, dim=(-2, -1))

    filters = torch.abs(rooted_filters)**2
    filters = filters.unsqueeze(0).unsqueeze(0)
    x = physics(img_tensor, filters=filters)

    # compute fidelity loss
    loss_fidel = criterion(x, y)

    # compute regularization term
    grad_h, grad_v = grad(filters.view(num_kernels_y, num_kernels_x, *filters.shape[-2:]), dim=(0, 1))
    R1 = grad_h.pow(2).sum() + grad_v.pow(2).sum()
    R2 = torch.sum((torch.sum(filters, dim=(0,1,3,4)) - 1).pow(2))
    loss_reg = alpha * R1 + (1 - alpha) * R2

    # total loss
    L = loss_fidel + lamb * loss_reg
    losses.append(L.item())
    
    optimizer.zero_grad()
    L.backward()
    optimizer.step()

    if i % 500 == 0 or i == n_iter - 1:
        print(f"iter {i:03d}: loss_fidel={loss_fidel.item():.6f}, loss_reg={loss_reg.item():.6f}, total={L.item():.6f}")
        x_deconv = physics.A_adjoint(y, filters=filters)
        show_images([img_tensor, x_deconv], title=["Original", "estimated"])
#%%
import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()

#%%
# show the gt and estimated pupil functions
for i in range(10):
    show_images([filters_gt[0, 0, i:i+1].unsqueeze(0), 
                 filters[0, 0, i:i+1].unsqueeze(0)],
                 title=[f"GT Filter {i}", f"Estimated Filter {i}"])


# %%
initial_lr = 1e-2
min_lr = 1e-4
n_iter_u = 2000
# u = physics.A_adjoint(y, estimated_filters).detach().requires_grad_(True)
padding = (15,15,15,15)
u = F.pad(y, padding).detach().requires_grad_(True)
optimizer_u = torch.optim.AdamW([u], lr=initial_lr)
lr_scheduler = CosineAnnealingLR(optimizer_u, eta_min=min_lr, T_max=n_iter_u)

estimated_filters = filters.detach()
metric = TotalVariation().to(**kwargs)
losses_u = []
for i in range(n_iter_u):
    u_blur = physics(u, filters=estimated_filters)
    loss_u = criterion(u_blur, y) + 0.001 * metric(u)
    # loss_u = criterion(u_blur, y)
    optimizer_u.zero_grad()
    loss_u.backward()
    optimizer_u.step()
    lr_scheduler.step()
    losses_u.append(loss_u.item())
    if i % 200 == 0 or i == n_iter_u - 1:
        print(f"iter {i:03d}: loss_u={loss_u.item():.6f}")
        show_images([img_tensor, u], title=["Original", "Recovered with estimated filters"]) 
# %%
# crop the center of the estimated image and the original image for better visualization
fig = plt.figure()
plt.plot(losses_u)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Loss curve for image recovery with estimated filters")
plt.show()

psnr = PeakSignalNoiseRatio(data_range=1.0).to(**kwargs)

crop_size = y.shape[-2:]
crop_u = u[..., (u.shape[-2] - crop_size[0]) // 2 : (u.shape[-2] + crop_size[0]) // 2, (u.shape[-1] - crop_size[1]) // 2 : (u.shape[-1] + crop_size[1]) // 2]
img_tensor_crop = img_tensor[..., (img_tensor.shape[-2] - crop_size[0]) // 2 : (img_tensor.shape[-2] + crop_size[0]) // 2, (img_tensor.shape[-1] - crop_size[1]) // 2 : (img_tensor.shape[-1] + crop_size[1]) // 2]
show_images([img_tensor_crop, crop_u, y],
            title=["Original", "Recovered", "Noisy"],
            suptitle=f'psnr={psnr(crop_u, img_tensor_crop).item():.2f}dB')
# %%
