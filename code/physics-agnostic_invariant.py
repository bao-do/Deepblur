#%%
import torch
import matplotlib.pyplot as plt
from putils import open_image, show_images
import os
import torch.fft as fft
import deepinv as dinv
from torchvision.utils import save_image
from objectives_function import LossFidelity
from deepinv.physics.generator import DiffractionBlurGenerator
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
kwargs = {'device': device, 'dtype': dtype}
absolute_path = os.path.abspath(os.path.dirname(__file__))
img_size = (256, 256)
psf_size = (31, 31)

img = open_image(os.path.join(absolute_path, "data/first_img.JPEG"), 
                 img_size=img_size,
                 **kwargs)
img_crop = img[..., psf_size[0]//2:-(psf_size[0]//2), psf_size[1]//2:-(psf_size[1]//2)]
show_images([img], title=["Original Image"])
# %%

kernel_generator = DiffractionBlurGenerator(psf_size=psf_size,
                                            num_channels=1,
                                            **kwargs)

blur = kernel_generator.step(batch_size=1)
kernel = blur['filter']
pupil = blur['pupil']
show_images([torch.real(pupil)], title=["Pupil Function"])
show_images([kernel], title=["Original Kernel"])
#%%
def blur_fn(x, kernel):
    kh, kw = kernel.shape[-2:]
    kernel_padded = torch.zeros_like(x)
    kernel_padded[:, :, :kh, :kw] = kernel
    kernel_padded = torch.roll(kernel_padded, shifts=(-(kh//2), -(kw//2)), dims=(-2, -1))
    x_fft = fft.fft2(x)
    kernel_fft = fft.fft2(kernel_padded)
    y_fft = x_fft * kernel_fft
    y = torch.real(fft.ifft2(y_fft)[...,kh//2:-(kh//2), kw//2:-(kw//2)])
    return y

y = blur_fn(img, kernel)
show_images([y], title=["Blurred Image"])

# %%
sigma_list = torch.tensor([0, 0.01, 0.05, 0.1], **kwargs)
# sigma_list = torch.tensor([0], **kwargs)

ys = y.expand(len(sigma_list), -1, -1, -1)
ys = ys + sigma_list.view(-1, 1, 1, 1) * torch.randn_like(y)
show_images(ys, title=[rf"$\sigma={sigma.item():.2f}$" for sigma in sigma_list])


# %%
# softmax = torch.nn.Softmax(dim=-1)
relu = torch.nn.SiLU()
def objective_fn(coeffs, blurr_true, norm="l2"):
    # h,w = coeffs.shape[-2:]
    kernel_est = relu(coeffs)
    kernel_est = kernel_est / kernel_est.sum()
    blur_x = blur_fn(img, kernel_est)
    if norm == "l2":
        return (blur_x - blurr_true).pow(2).sum()
    elif norm == "l1":
        return (blur_x - blurr_true).abs().sum()
    
def initial_coeffs(psf_size, x, y):
    fft_x = fft.fft2(x)
    fft_y = fft.fft2(y)
    fft_h = fft_y / (fft_x + 1e-8)
    h = torch.real(fft.ifft2(fft_h))
    h = torch.roll(h, shifts=(psf_size[0]//2, psf_size[1]//2), dims=(-2, -1))
    h_cropped = h[..., :psf_size[0], :psf_size[1]]
    return h_cropped
#%%
niter = 20000
learning_rate = 1e-2
eta_min = 1e-6




for i, blur_true in enumerate(ys):
    print("#######################################################################")
    print(f"Starting optimization for sigma={sigma_list[i].item():.2f}")

    coeffs = initial_coeffs(psf_size, img_crop, blur_true)
    # coeffs = torch.randn(1,1, *psf_size, **kwargs)
    coeffs = coeffs.requires_grad_(True)

    optimizer = Adam([coeffs], lr=learning_rate)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=niter, eta_min=eta_min)
    # lr_scheduler = ExponentialLR(optimizer, gamma=0.996)

    kernel_iter = []
    loss_iter = []
    iter = []
    for i in range(niter):
        loss = objective_fn(coeffs, blur_true.unsqueeze(0), norm="l1")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        if (i % 200 == 0) or (i+1 == niter):
            print(f"Iteration {i+1}/{niter}, Loss: {loss.item()}")
            # kernel_est = softmax(coeffs.flatten()).view(*psf_size).detach()
            kernel_est = relu(coeffs)
            kernel_est = kernel_est / kernel_est.sum()
            kernel_iter.append(kernel_est.detach())
            loss_iter.append(loss.item())
            iter.append(i+1)
    
    kernel_iter = torch.cat(kernel_iter)
    show_images([kernel, kernel_est],
                title=["Original Kernel", "Estimated Kernel"])
    show_images(kernel_iter,
                title=[rf"Iter {it}" for it in iter],
                ncols=5)
    

# %%
