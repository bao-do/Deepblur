#%%
import torch
import matplotlib.pyplot as plt
from putils import open_image, show_images
import os
import torch.fft as fft
import deepinv as dinv
from torchvision.utils import save_image


# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
kwargs = {'device': device, 'dtype': dtype}
absolute_path = os.path.abspath(os.path.dirname(__file__))
img_size = (256, 256)

figure_path = os.path.abspath(os.path.join(absolute_path, ".."))
figure_path = os.path.join(figure_path, "tex/figures/baseline_method")
os.makedirs(figure_path, exist_ok=True)
img = open_image(os.path.join(absolute_path, "data/first_img.JPEG"),
                 img_size=img_size,
                 **kwargs)

img = 10*torch.rand(1, 1, *img_size, **kwargs)
# save_image(img, os.path.join(figure_path, "original_image.png"))
# %%

psf_size = (31, 31)
blur = dinv.physics.generator.DiffractionBlurGenerator(psf_size=psf_size,
                                                       num_channels=1,
                                                       **kwargs)

kernel = blur.step(batch_size=1)['filter']
y = torch.nn.functional.conv2d(img, kernel.flip(dims=(-2, -1)), padding='valid')

# %%
sigma_list = torch.tensor([0, 0.01, 0.05, 0.1, 0.2, 0.5], **kwargs)

y = y.expand(len(sigma_list), -1, -1, -1)
y = y + sigma_list.view(-1, 1, 1, 1) * torch.randn_like(y)
show_images([img], title=["Original Image"]) 
show_images(y, title=[rf"$\sigma={sigma.item():.2f}$" for sigma in sigma_list])
# %% reverse kernel
padding_h = kernel.shape[-2]//2
padding_w = kernel.shape[-1]//2
img_crop = img[:, :, padding_h:-padding_h, padding_w:-padding_w]

kernel_est = fft.ifft2(fft.fft2(y)/fft.fft2(img_crop)).real
kernel_est = torch.roll(kernel_est, (padding_h, padding_w), dims=(-2,-1))
kernel_est = kernel_est[:, :, :kernel.shape[-2], :kernel.shape[-1]]
show_images([torch.cat([kernel, kernel_est], dim=0)],
            title=["Original Kernel"]+[rf"$\sigma={sigma.item():.2f}$" for sigma in sigma_list])


#%%
for i, sigma in enumerate(sigma_list):
    fig = plt.figure()
    plt.imshow(kernel_est[i].permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path,
                             f"estimated_kernel_sigma_{sigma.item():.2f}_noise.png"),
                bbox_inches="tight",
                pad_inches=0)
    
# %%
kernel_np = kernel[0].permute(1, 2, 0).cpu().numpy()
plt.imshow(kernel_np)
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(figure_path, "original_kernel_noise.png"),
            bbox_inches="tight",
            pad_inches=0)



# %%
