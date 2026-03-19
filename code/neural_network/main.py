import torch
import torch.nn as nn
from torch.optim import AdamW, LBFGS
from objectives_function import LossFidelity, blur_fn_invariant
from deepinv.physics.generator import DiffractionBlurGenerator
from tqdm import tqdm
from typing import Tuple
from deepinv.physics.functional import conv2d_fft
from warnings import warn



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x
    

class PsfCalibration(nn.Module):
    learning_rate = 5e-3
    eta_min = 1e-6
    psf_size = (31, 31)
    max_zernike_amplitude = 0.3
    pupil_size: Tuple[int, int]=(256, 256)
    input_dim = 384
    niter = 300
    objective_fn = LossFidelity
    T_max= 300

    def __init__(self, num_coeffs: int,
                 device: str='cpu',
                 dtype = torch.float32,
                 verbose: bool = True,
                 **kwargs):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.factorial_kwargs = dict(device=device, dtype=dtype)

        self.verbose = verbose

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
                           initial_coeffs: torch.Tensor = None,
                           niter: int= None,
                           crop: bool=True,
                           optimizer_type='lbfgs'):
        model = MLP(input_dim=self.input_dim,
                    output_dim=self.num_coeffs).to(**self.factorial_kwargs)
        # no scheduler for lbfgs

        if optimizer_type == 'adamw':
            optimizer = AdamW(model.parameters(),
                              lr=self.learning_rate,
                              weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                T_max=self.T_max,
                                                                eta_min=self.eta_min)
        else:
            if optimizer_type != 'lbfgs':
                warn(f"Unknown optimizer: {optimizer_type}, using LBFGS instead")
            optimizer = LBFGS(model.parameters(),
                                lr=1.0,
                                history_size=10,
                                max_iter=20,
                                line_search_fn="strong_wolfe")
        if initial_coeffs is not None:
            xi = initial_coeffs
        else:
            xi = torch.randn(1, self.input_dim, **self.factorial_kwargs)

        loss_iter = []
        niter = self.niter if niter is None else niter
        
        objective_fn = LossFidelity(reduction="sum",
                            norm="l2",
                            physics=conv2d_fft,
                            **self.factorial_kwargs)
        if self.verbose:
            progress = tqdm(range(niter), desc="Progressing")
        else:
            progress = range(niter)
        for i in progress:
            def closure():
                coeffs = model(xi)
                filters = self.kernel_generator.step(batch_size=1, coeff=coeffs)['filter']
                optimizer.zero_grad()
                loss = objective_fn(x, y, filter=filters, crop=crop)
                loss.backward()
                return loss

            optimizer.step(closure)

            with torch.no_grad():
                coeffs_post = model(xi)
                filters_post = self.kernel_generator.step(batch_size=1, coeff=coeffs_post)['filter']
                loss_val = objective_fn(x, y, filter=filters_post, crop=crop)
            loss_iter.append(loss_val.item())

            if optimizer_type == 'adamw':
                scheduler.step()

            if (i % 10 == 0) or (i == niter - 1):
                if self.verbose:
                    progress.set_postfix({"loss": loss_iter[-1]})

        blur = self.kernel_generator.step(batch_size=1,
                                        coeff=model(xi))
        return blur, loss_iter
