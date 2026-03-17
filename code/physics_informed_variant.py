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
generator = DiffractionBlurGenerator(kernel_size,
                                     channels,
                                     pupil_size=pupil_size,
                                     **kwargs)
print("\n".join(generator.zernike_polynomials)) # list of Zernike polynomials used
#%%
blur_gt = generator.step(batch_size=num_kernels)  # dict_keys(['filter', 'coeff', 'pupil'])
filters_gt = blur_gt['filter'].transpose(0, 1).unsqueeze(0)
coeffs_gt = blur_gt['coeff']

physics = TiledSpaceVaryingBlur(patch_size=patch_size, stride=stride, **kwargs)
y = physics(img_tensor, filters=filters_gt)
# %%
sigma_list = torch.tensor([0, 0.01, 0.05, 0.1], **kwargs)

sigma = 0.0

ys = y +  sigma_list.view(-1,1,1,1) * torch.randn_like(y)
show_images(ys, title=[rf"$\sigma={sigma.item():.2f}$" for sigma in sigma_list])
# %%

totalloss = TotalLoss(kernel_size=kernel_size,
                      num_kernels=(num_kernels_y, num_kernels_x),
                      filters_reg_coeffs=(0.1, 0, 0),
                      r=3,
                      coeffs=(0.1, 0),
                      physics=physics,
                      basis="zernike",
                      filters_generator=generator,
                      **factorial_kwargs)
#%%
n_iter = 60
losses = []
n_restarts = 5
learning_rate = 1.0

results = {}


for sigma, blur_true in zip(sigma_list, ys):
    print("#######################################################################")
    print(f"Starting optimization for sigma={sigma.item():.2f}")
    best_loss = float('inf')
    best_loss_iter = None
    best_kernel_est = None

    for restart in range(n_restarts):

        sample = generator.step(batch_size=num_kernels,
                            seed=torch.randint(0, 10000, (1,)).item())
        coeffs = sample['coeff'].requires_grad_(True)
        kernel = sample['filter']

        optimizer = torch.optim.LBFGS([coeffs], max_iter=20, line_search_fn="strong_wolfe", lr=learning_rate)

        print(f"Restart {restart+1}/{n_restarts}")
        loss_iter = []
        for i in range(n_iter):
            loss = None

            def closure():
                global loss
                optimizer.zero_grad()
                Ls = totalloss.forward(img_tensor, blur_true.unsqueeze(0), projection_coeffs=coeffs)
                L = torch.sum(Ls)
                loss = L.item()
                L.backward()
                return L

            optimizer.step(closure)
            loss_iter.append(loss)

            if i % 10 == 0 or i == n_iter - 1:
                print(f"iter {i:03d}: loss {loss:.6f}")
                filters_est = generator.step(batch_size=num_kernels, coeff=coeffs)['filter']
                random_indices = torch.randint(0, num_kernels, (5,))
                # show_images(filters_est[random_indices],
                #             ncols=num_kernels_x, 
                #             suptitle=f"Estimated filters at iter {i}")
                # show_images(filters_gt[0,0][random_indices].unsqueeze(1),
                #             ncols=num_kernels_x,
                #             suptitle=f"GT filters")

        if loss_iter[-1] < best_loss:
            best_loss = loss_iter[-1]
            best_loss_iter = loss_iter
            best_kernel_est = generator.step(batch_size=num_kernels, coeff=coeffs)['filter'].detach()

    random_indices = torch.randint(0, num_kernels, (5,))
    show_images(best_kernel_est[random_indices],
                ncols=num_kernels_x,
                suptitle=f"Best estimated filters after {n_iter} iterations")
    show_images(filters_gt[0,0][random_indices].unsqueeze(1),
                ncols=num_kernels_x,
                suptitle=f"GT filters")

    fig = plt.figure(figsize=(10, 5))
    plt.plot(best_loss_iter)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Best Loss Curve over {n_iter} iterations")
    plt.show()

    results[sigma.item()] = {
        "best_loss": best_loss,
        "best_loss_iter": best_loss_iter,
        "best_kernel_est": best_kernel_est
    }



# %%
