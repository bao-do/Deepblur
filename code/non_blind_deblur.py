#%%
import os
project_abs_path = os.path.abspath(os.path.dirname(__file__)) 
import torch
import deepinv as dinv
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

img_size = (256, 256)
patch_size = (64, 64)
stride = (32, 32)

kernel_size = (31,31)
num_kernels_x = 7
num_kernels_y = 7
num_kernels = num_kernels_x * num_kernels_y 
channels = 1
pupil_size = (256, 256)

img_tensor = open_image(os.path.join(project_abs_path, "data/first_img.JPEG"),
                        img_size=img_size,
                        **kwargs)
show_images(img_tensor)
# %% Generate diffraction blur kernels
generator = DiffractionBlurGenerator(kernel_size, channels, pupil_size=pupil_size, **kwargs)
print("\n".join(generator.zernike_polynomials)) # list of Zernike polynomials used
#%%
blur_gt = generator.step(batch_size=num_kernels)  # dict_keys(['filter', 'coeff', 'pupil'])
filters_gt = blur_gt['filter'].transpose(0, 1).unsqueeze(0)
coeffs_gt = blur_gt['coeff']

# %%
sigma = 0.01
physics = TiledSpaceVaryingBlur(patch_size=patch_size, stride=stride, **kwargs)
y = physics(img_tensor, filters=filters_gt)

y += sigma * torch.randn_like(y)
print(img_tensor.shape, y.shape)
dinv.utils.plot([img_tensor, y], titles=["Original", "Blurred"])

# %%
# initialize coeffs with random values
generator = DiffractionBlurGenerator(kernel_size,
                                     channels,
                                     pupil_size=pupil_size,
                                     **kwargs)
coeffs = generator.generate_coeff(batch_size=num_kernels).requires_grad_(True)
print(coeffs.requires_grad)
kernel = generator.step(batch_size=num_kernels, coeff=coeffs)['filter']
print(kernel.requires_grad)



learning_rate = 0.01
optimizer = torch.optim.AdamW([coeffs], lr=learning_rate)

#%%
totalloss = TotalLoss(kernel_size=kernel_size,
                      num_kernels=(num_kernels_y, num_kernels_x),
                      filters_reg_coeffs=(0.5, 0.5, 0),
                      r=3,
                      coeffs=(0.1, 0),
                      physics=physics,
                      basis="zernike",
                      filters_generator=generator,
                      **factorial_kwargs)
#%%
n_iter = 500
losses = []
for i in range(n_iter):
    Ls = totalloss.forward(img_tensor, y, projection_coeffs=coeffs)
    L = torch.sum(Ls)
    
    losses.append(L.item())
    
    optimizer.zero_grad()
    L.backward()
    optimizer.step()

    if i % 100 == 0 or i == n_iter - 1:
        print(f"iter {i:03d}: loss_fidel={Ls[0].item():.6f}, loss_reg={Ls[1].item()+Ls[2].item():.6f}, total={L.item():.6f}")
#%%
import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()


# %%
estimated_blur = generator.step(batch_size=num_kernels, coeff=coeffs)
estimated_filters = estimated_blur['filter'].transpose(0, 1).unsqueeze(0)
show_images([physics(img_tensor, filters=estimated_filters), y], title=["Blurred with estimated filters", "Blurred with GT filters"])
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
# approximate the pupil function with fourier atoms

# frequency grid for a ball of radius r in the Fourier domain
n = 10

coeffs_fourier = torch.randn((num_kernels, 1, 2*n+1, 2*n+1),
                             dtype= torch.complex64,
                             device=device)*0.01

coeffs_fourier = coeffs_fourier.requires_grad_(True)

#%%
totalloss = TotalLoss(kernel_size=2*n+1,
                      num_kernels=(num_kernels_y, num_kernels_x),
                      filters_reg_coeffs=(0.5, 0.5, 0.002),
                      r=2,
                      coeffs=(0.1, 0),
                      physics=physics,
                      basis="fourier",
                      filters_generator=generator,
                      **factorial_kwargs)

#%%
n_iter = 4000
losses = []

learning_rate = 0.01
optimizer = torch.optim.AdamW([coeffs_fourier], lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=n_iter, eta_min=1e-3)


for i in range(n_iter):
    Ls = totalloss.forward(img_tensor, y, projection_coeffs=coeffs_fourier)
    L = torch.sum(Ls)
    
    losses.append(L.item())
    
    optimizer.zero_grad()
    L.backward()
    optimizer.step()
    scheduler.step()

    if i % 500 == 0 or i == n_iter - 1:
        print(f"iter {i:03d}: loss_fidel={Ls[0].item():.6f}, loss_reg={Ls[1].item()+Ls[2].item():.6f}, total={L.item():.6f}")
#%%
import matplotlib.pyplot as plt
plt.plot(losses)
plt.yscale('log')
plt.show()

#%%
# show the gt and estimated pupil functions
filter = totalloss.filters_generator(coeffs_fourier)
# random 10 indices
indices = torch.randperm(num_kernels)[:10]
show_images(filters_gt[0,0,indices,5:-5,5:-5].unsqueeze(1),
            suptitle="GT Filters")
show_images(filter[indices,...],
            suptitle="Estimated Filters") 




# %%
