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

absolute_path = os.path.abspath(os.path.dirname(__file__))

torch.manual_seed(42)


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

kernel_generator = DiffractionBlurGenerator(psf_size=psf_size,
                                            max_zernike_amplitude=max_zernike_amplitude,
                                            zernike_index=range(2, 12),
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
# sigma_list = torch.tensor([0, 0.01, 0.05, 0.1], **kwargs)
sigma_list = torch.tensor([0], **kwargs)

ys = y.expand(len(sigma_list), -1, -1, -1)
ys = ys + sigma_list.view(-1, 1, 1, 1) * torch.randn_like(y)
show_images(ys, title=[rf"$\sigma={sigma.item():.2f}$" for sigma in sigma_list])
# %%

objective_fn = LossFidelity(reduction="sum",
                            norm="l1",
                            physics=blur_fn_invariant,
                            **kwargs)



#%%
niter = 300
learning_rate = 5e-3
eta_min = 1e-6
input_dim = 256
output_dim = 15

kernel_est_generator = DiffractionBlurGenerator(psf_size=psf_size,
                                                max_zernike_amplitude=0.3,
                                                zernike_index=range(2, output_dim+2),
                                                num_channels=1,
                                                **kwargs)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x


best_kernel_est_list = []
for sigma, blur_true in zip(sigma_list, ys):

    xi = torch.randn(1,input_dim, **kwargs)
    model = MLP(input_dim=input_dim, output_dim=output_dim).to(**kwargs)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=niter, eta_min=eta_min)
 
    loss_iter = []
    progress = tqdm(range(niter), desc=f"Optimizing for sigma={sigma.item():.2f}")
    for i in progress:
        coeffs = model(xi)
        filters = kernel_est_generator.step(batch_size=1,
                                        coeff=coeffs)['filter']
        optimizer.zero_grad()
        loss = objective_fn(img, blur_true.unsqueeze(0), filters=filters, crop=False)
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

    # show spectrum 
    show_images([fft.fft2(kernel).abs(),
                 fft.fft2(kernel_est).abs()])


            

# %%

class PsfCalibration(nn.Module):
    learning_rate = 5e-3
    eta_min = 1e-6
    psf_size = (31, 31)
    max_zernike_amplitude = 0.3
    pupil_size: Tuple[int, int]=(256, 256)
    input_dim = 256
    niter = 300
    objective_fn = LossFidelity
    T_max= 300

    def __init__(self, num_coeffs: int,
                 device: str='cpu',
                 dtype = torch.float32,
                 **kwargs):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.factorial_kwargs = dict(device=device, dtype=dtype)

        self.num_coeffs = min(31, num_coeffs)

        self.kernel_generator = DiffractionBlurGenerator(psf_size=self.psf_size,
                                            max_zernike_amplitude=self.max_zernike_amplitude,
                                            zernike_index=range(2, self.num_coeffs+2),
                                            num_channels=1,
                                            **self.factorial_kwargs)
        
        for key, value in kwargs.items():
            setattr(self, key, value)

    
    def update_parameters(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _forward_one_image(self,
                           x: torch.Tensor,
                           y: torch.Tensor,
                           niter: int= None,
                           crop: bool=True):
        model = MLP(input_dim=self.input_dim,
                    output_dim=self.num_coeffs).to(**self.factorial_kwargs)
        optimizer = AdamW(model.parameters(),
                          lr=self.learning_rate,
                          weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=self.T_max,
                                                               eta_min=self.eta_min)
        xi = torch.randn(1, self.input_dim, **self.factorial_kwargs)

        loss_iter = []
        niter = self.niter if niter is None else niter
        
        objective_fn = LossFidelity(reduction="sum",
                            norm="l2",
                            physics=blur_fn_invariant,
                            **self.factorial_kwargs)
        
        progress = tqdm(range(niter), desc="Progressing")
        for i in progress:
            coeffs = model(xi)
            filters = self.kernel_generator.step(batch_size=1,
                                            coeff=coeffs)['filter']
            optimizer.zero_grad()
            loss = objective_fn(x, y, filters=filters, crop=crop)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_iter.append(loss.item())
            if (i % 10 == 0) or (i == niter - 1):
                progress.set_postfix({"loss": loss.item()})      
        
        blur = self.kernel_generator.step(batch_size=1,
                                        coeff=model(xi))
        return blur, loss_iter

#%%

# class PsfCalibration(nn.Module):
#     learning_rate = learning_rate
#     eta_min = eta_min
#     psf_size = psf_size
#     max_zernike_amplitude = 0.3
#     input_dim = input_dim
#     niter = niter
#     T_max= niter

#     def __init__(self, num_coeffs: int,
#                  device: str='cpu',
#                  dtype = torch.float32,
#                  **kwargs):
#         super().__init__()
#         self.device = device
#         self.dtype = dtype
#         self.factorial_kwargs = dict(device=device, dtype=dtype)

#         self.num_coeffs = min(31, num_coeffs)

#         # self.kernel_generator = DiffractionBlurGenerator(psf_size=self.psf_size,
#         #                                     max_zernike_amplitude=self.max_zernike_amplitude,
#         #                                     zernike_index=range(2, output_dim+2),
#         #                                     num_channels=1,
#         #                                     **self.factorial_kwargs)
#         self.kernel_generator = kernel_est_generator
        
#         for key, value in kwargs.items():
#             setattr(self, key, value)

    
#     def update_parameters(self, **kwargs):
#         for key, value in kwargs.items():
#             setattr(self, key, value)

#     def _forward_one_image(self,
#                            x: torch.Tensor,
#                            y: torch.Tensor,
#                            niter: int= None,
#                            crop: bool=True):
#         xi = torch.randn(1, input_dim, **self.factorial_kwargs)
#         model = MLP(input_dim=self.input_dim,
#                     output_dim=output_dim).to(**self.factorial_kwargs)
#         optimizer = AdamW(model.parameters(),
#                           lr=self.learning_rate,
#                           weight_decay=1e-4)
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
#                                                                T_max=self.T_max,
#                                                                eta_min=self.eta_min)

#         loss_iter = []
#         niter = self.niter if niter is None else niter
        
#         # objective_fn = LossFidelity(reduction="sum",
#         #                     norm="l2",
#         #                     physics=blur_fn_invariant,
#         #                     **self.factorial_kwargs)
        
#         progress = tqdm(range(niter), desc="Progressing")
#         for i in progress:
#             coeffs = model(xi)
#             filters = self.kernel_generator.step(batch_size=1,
#                                             coeff=coeffs)['filter']
#             optimizer.zero_grad()
#             loss = objective_fn(x, y, filters=filters, crop=False)
#             loss.backward()
#             optimizer.step()
#             scheduler.step()

#             loss_iter.append(loss.item())
#             if (i % 10 == 0) or (i == niter - 1):
#                 progress.set_postfix({"loss": loss.item()})      
        
#         blur = self.kernel_generator.step(batch_size=1,
#                                         coeff=model(xi))
#         return blur, loss_iter
#%%
psfcalib = PsfCalibration(num_coeffs=15, **kwargs)
blur_est, loss_iter = psfcalib._forward_one_image(img,
                                                  ys[:1],
                                                  niter=300,
                                                  crop=False)
show_images([kernel, blur_est['filter']],
            title=["Original Kernel", "Estimated Kernel"],
            suptitle=f'relative error: {torch.norm(kernel-blur_est["filter"])/torch.norm(kernel):.4f}')
fig = plt.figure(figsize=(10, 5))
plt.plot(loss_iter)g
plt.xlabel("Iteration")
plt.ylabel("Loss")  
plt.title(f"Loss Curve")
# %%
