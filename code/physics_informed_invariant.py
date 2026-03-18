#%%
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
save_image(img, os.path.join(figure_path, "original_image.png"))
# %%
max_zernike_amplitude = 0.3

kernel_generator = DiffractionBlurGenerator(psf_size=psf_size,
                                            max_zernike_amplitude=max_zernike_amplitude,
                                            num_channels=1,
                                            **kwargs)
def random_seed():
    return torch.randint(0, 10000, (1,)).item()

blur = kernel_generator.step(batch_size=1, seed=random_seed())
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
# %%

objective_fn = LossFidelity(reduction="sum",
                            norm="l2",
                            physics=blur_fn_invariant,
                            **kwargs)


kernel_est_generator = DiffractionBlurGenerator(psf_size=psf_size,
                                            max_zernike_amplitude=0.3,
                                            zernike_index=range(2,17 ),
                                            num_channels=1,
                                            **kwargs)
#%%
niter = 10
learning_rate = 1e-3
eta_min = 1e-8

n_restarts = 5

best_kernel_est_list = []
for sigma, blur_true in zip(sigma_list, ys):
    best_loss = float('inf')
    best_loss_iter = None
    best_kernel_est = None
    
    for restart in range(n_restarts):
        print("#######################################################################")
        print(f"Restart {restart+1}/{n_restarts}: optimization for sigma={sigma.item():.2f}")

        coeffs = kernel_est_generator.step(batch_size=1,
                                        seed=random_seed())['coeff'].requires_grad_(True)
        # coeffs = torch.zeros(1,8, **kwargs)
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
                loss = objective_fn(img, blur_true.unsqueeze(0), filters=filters, crop=False)
                loss.backward()
                return loss
            optimizer.step(closure)
            loss = closure()
            loss_iter.append(loss.item())
            if (i % 10 == 0) or (i == niter - 1):
                print(f"Iteration {i+1}/{niter}, Loss: {loss.item():.10f}")
        
        kernel_est = kernel_est_generator.step(batch_size=1,
                                            coeff=coeffs)['filter'].detach()
        # show_images([kernel, kernel_est],
        #             title=["Original Kernel", "Estimated Kernel"],
        #             suptitle=f"relative error: {torch.norm(kernel-kernel_est)/torch.norm(kernel):.4f}")


        if loss_iter[-1] < best_loss:
            best_loss = loss_iter[-1]
            best_loss_iter = loss_iter
            best_kernel_est = kernel_est_generator.step(batch_size=1,
                                                    coeff=coeffs)['filter'].detach()

    # fig = plt.figure(figsize=(10, 5))
    # plt.plot(loss_iter)
    # plt.xlabel("Iteration")
    # plt.ylabel("Loss")
    # plt.title(f"Loss Curve for sigma={sigma.item():.2f}")
    # plt.show()

    show_images([kernel, best_kernel_est],
                title=["Original Kernel", "Estimated Kernel"],
                suptitle=f"relative error: {torch.norm(kernel-best_kernel_est)/torch.norm(kernel):.4f}")
    show_images([torch.log(torch.abs(fft.fftshift(fft.fft2(kernel)))+1e-8),
            torch.log(torch.abs(fft.fftshift(fft.fft2(best_kernel_est)))+1e-8)],
            title=["Original Kernel Spectrum", "Estimated Kernel Spectrum"])

    best_kernel_est_list.append(best_kernel_est)

    

# %%
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(5, 5))
plt.imshow(kernel[0,0].cpu().numpy(), cmap='viridis')
plt.axis('off')
plt.savefig(os.path.join(figure_path, f"original_kernel.png"),
            bbox_inches="tight",
            pad_inches=0)

for sigma, kernel_est in zip(sigma_list, best_kernel_est_list):
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(kernel_est[0,0].cpu().numpy(), cmap='viridis')
    plt.axis('off')
    plt.savefig(os.path.join(figure_path, f"estimated_kernel_sigma_{sigma.item():.2f}.png"),
                bbox_inches="tight",
                pad_inches=0)
# %%
