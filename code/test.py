#%%
from  neural_network import MLP
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
max_zernike_amplitude = 0.2

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
# sigma_list = torch.tensor([0.01], **kwargs)

ys = y.expand(len(sigma_list), -1, -1, -1)
ys = ys + sigma_list.view(-1, 1, 1, 1) * torch.randn_like(y)
show_images(ys, title=[rf"$\sigma={sigma.item():.2f}$" for sigma in sigma_list])
# %%

objective_fn = LossFidelity(reduction="sum",
                            norm="l2",
                            physics=blur_fn_invariant,
                            **kwargs)



#%%
niter = 150
learning_rate = 5e-3
eta_min = 1e-6
input_dim = 256
output_dim = 15

kernel_est_generator = DiffractionBlurGenerator(psf_size=psf_size,
                                                max_zernike_amplitude=3,
                                                zernike_index=range(1, output_dim+1),
                                                num_channels=1,
                                                **kwargs)


best_kernel_est_list = []
for sigma, blur_true in zip(sigma_list, ys):

    xi = torch.zeros(1,input_dim, **kwargs)
    model = MLP(input_dim=input_dim, output_dim=output_dim).to(**kwargs)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=niter, eta_min=eta_min)
 
    loss_iter = []
    progress = tqdm(range(niter), desc=f"Optimizing for sigma={sigma.item():.2f}")
    for i in progress:
        coeffs = model(xi)
        filters = kernel_generator.step(batch_size=1,
                                        coeff=coeffs)['filter']
        optimizer.zero_grad()
        loss = objective_fn(img, blur_true.unsqueeze(0), filters=filters)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_iter.append(loss.item())
        if (i % 10 == 0) or (i == niter - 1):
            progress.set_postfix({"loss": loss.item()})
            


    fig = plt.figure(figsize=(10, 5))
    plt.plot(loss_iter)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve for sigma={sigma.item():.2f}")
    plt.show()

    coeffs = model(xi)
    kernel_est = kernel_generator.step(batch_size=1,
                                        coeff=coeffs)['filter']
    show_images([kernel, kernel_est],
                title=["Original Kernel", "Estimated Kernel"],
                suptitle=f'relative error: {torch.norm(kernel-kernel_est)/torch.norm(kernel):.4f}')



            

# %%
import torch.nn as nn
from typing import Tuple
class PsfCalibration(nn.Module):
    learning_rate = 5e-3
    eta_min = 1e-6
    psf_size = (31, 31)
    max_zernike_amplitude = 0.3
    pupil_size: Tuple[int, int]=(256, 256)
    input_dim = 256
    niter = 150

    def __init__(self, num_coeffs: int,
                 device: str='cpu',
                 model: nn.Module = None,
                 **kwargs):
        super().__init__()
        self.num_coeffs = num_coeffs
        if model is None:
            self.model = MLP(input_dim=self.input_dim, output_dim=num_coeffs).to(device)
        else:
            self.model = model.to(device)
        self.device = device

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.optimizer = AdamW(self.model.parameters(),
                               lr=self.learning_rate,
                               weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                               T_max=self.niter,
                                                               eta_min=self.eta_min)
    
    def update_parameters(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _forward_one_image(x: torch.Tensor,
                           y: torch.Tensor,
                           kernel_generator: DiffractionBlurGenerator,
                           objective_fn: LossFidelity):
        
