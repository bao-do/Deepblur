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
from torch.optim import Adam, LBFGS

absolute_path = os.path.abspath(os.path.dirname(__file__))
figure_path = os.path.abspath(os.path.join(absolute_path, ".."))
figure_path = os.path.join(figure_path, "tex/figures/physics_agnostic_invariant")

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
from warnings import warn
softmax = torch.nn.Softmax(dim=-1)
silu = torch.nn.SiLU()
relu = torch.nn.ReLU()

def parameterize_kernel(coeffs, parameterization="softmax"):
    if parameterization == "softmax":
        kernel_est = softmax(coeffs.flatten()).view(*psf_size)
    elif parameterization == "relu":
        kernel_est = relu(coeffs)
        kernel_est = kernel_est / kernel_est.sum()
    elif parameterization == "silu":
        kernel_est = silu(coeffs)
        kernel_est = kernel_est / kernel_est.sum()
    else:
        warn(f"Unsupported parameterization: {parameterization}, using softmax by default.")
        kernel_est = softmax(coeffs.flatten()).view(*psf_size)
    return kernel_est

def objective_fn(coeffs, blurr_true, norm="l2", parameterization="softmax"):
    kernel_est = parameterize_kernel(coeffs, parameterization=parameterization)
    blur_x = blur_fn(img, kernel_est)
    L = 0
    if norm == "l2":
        L = (blur_x - blurr_true).pow(2).sum()
    elif norm == "l1":
        L = (blur_x - blurr_true).abs().sum()
    return L
    
def initial_coeffs(psf_size, x, y):
    fft_x = fft.fft2(x)
    fft_y = fft.fft2(y)
    fft_h = fft_y / (fft_x + 1e-8)
    h = torch.real(fft.ifft2(fft_h))
    h = torch.roll(h, shifts=(psf_size[0]//2, psf_size[1]//2), dims=(-2, -1))
    h_cropped = h[..., :psf_size[0], :psf_size[1]]
    return h_cropped

def closure():
    optimizer.zero_grad()
    loss = objective_fn(coeffs,
                        blur_true.unsqueeze(0),
                        norm="l2",
                        parameterization="softmax") 
    loss.backward()
    return loss
#%%
niter = 300
learning_rate = 1e-2
eta_min = 1e-4
n_restarts = 5



best_kernel_est_list = []
for sigma, blur_true in zip(sigma_list,ys):
    best_loss = float('inf')
    best_loss_iter = None
    best_kernel_est = None
    print("#######################################################################")
    print(f"Starting optimization for sigma={sigma.item():.2f}")
    for restart in range(n_restarts):
        print(f"Restart {restart+1}/{n_restarts}")
        coeffs = torch.randn(1,1, *psf_size, **kwargs)
        coeffs = coeffs.requires_grad_(True)

        optimizer = LBFGS([coeffs],
                          lr=learning_rate,
                          max_iter=20,
                          line_search_fn='strong_wolfe')

        kernel_iter = []
        loss_iter = []
        iter = []
        loss_iter = []
        for i in range(niter):
            optimizer.step(closure)
            loss = closure()
            loss_iter.append(loss.item())
            
            if (i % 10 == 0) or (i+1 == niter):
                print(f"Iteration {i+1}/{niter}, Loss: {loss.item()}")


        if loss_iter[-1] < best_loss:
            best_loss = loss_iter[-1]
            best_loss_iter = loss_iter
            kernel_est = softmax(coeffs.flatten()).view(1,*psf_size).detach()
            best_kernel_est = kernel_est
        
    fig = plt.figure(figsize=(10, 5))
    plt.plot(loss_iter)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve for sigma={sigma.item():.2f}")
    plt.show()

    show_images([kernel, best_kernel_est.unsqueeze(0)],
                title=["Original Kernel", "Estimated Kernel"])
    show_images([torch.log(torch.abs(fft.fftshift(fft.fft2(kernel)))+1e-8),
            torch.log(torch.abs(fft.fftshift(fft.fft2(best_kernel_est)))+1e-8)],
            title=["Original Kernel Spectrum", "Estimated Kernel Spectrum"])

    best_kernel_est_list.append(best_kernel_est)
    

# %%
# Calculate the hessian of the loss function at the estimated kernel
# in 1D for simplicity
from torch.autograd.functional import hessian
from functools import partial

def blur_fn_1d(x, kernel):
    kh = kernel.shape[-1]
    kernel_padded = torch.zeros_like(x)
    kernel_padded[..., :kh] = kernel
    kernel_padded = torch.roll(kernel_padded, shifts=-(kh//2), dims=-1)
    x_fft = fft.fft(x)
    kernel_fft = fft.fft(kernel_padded)
    y_fft = x_fft * kernel_fft
    y = torch.real(fft.ifft(y_fft)[..., kh//2:-(kh//2)])
    return y

def objective_fn_1d(coeffs, blurr_true, norm="l2"):
    # kernel_est = softmax(coeffs.flatten())
    # kernel_est = relu(coeffs)
    # kernel_est = kernel_est / kernel_est.sum()
    # blur_x = blur_fn_1d(img, kernel_est)
    blur_x = blur_fn_1d(img, coeffs)
    L = 0
    if norm == "l2":
        L = (blur_x - blurr_true).pow(2).sum()
    elif norm == "l1":
        L = (blur_x - blurr_true).abs().sum()
    return L
# %%
blur_true = ys[0,0,0,:]
coeffs_1d = torch.randn(psf_size[0], **kwargs)

objective_fn_1d_partial = partial(objective_fn_1d, blurr_true=blur_true, norm="l2")
hess = hessian(objective_fn_1d_partial, coeffs_1d)
hess_eigvals = torch.linalg.eigvals(hess)
plt.plot(hess_eigvals.real.cpu().numpy())
plt.xlabel("Real Part of Eigenvalues")
plt.ylabel("Imaginary Part of Eigenvalues")
plt.title("Eigenvalues of the Hessian at the Estimated Kernel")
plt.grid()
plt.show()


# %%
