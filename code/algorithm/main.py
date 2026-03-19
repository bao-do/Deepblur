import sys, os
absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(absolute_path)

from objectives_function import LossFidelity
import torch
from torch.optim import  LBFGS
from deepinv.physics.generator import DiffractionBlurGenerator
from deepinv.physics.functional import conv2d_fft
from putils import random_seed
from warnings import warn
from torch.quasirandom import SobolEngine


class LbfgsPsfCalibration(torch.nn.Module):

    def __init__(self,
                 device: str='cpu',
                 dtype = torch.float32,
                 psf_size: tuple = (81,81),
                 fc: float = 0.15,
                 num_coeffs: int = 33,
                 max_zernike_amplitude: float = 0.3,
                 **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.dtype = dtype
        self.factorial_kwargs = dict(device=device, dtype=dtype)

        self.psf_size = psf_size
        self.fc = fc
        self.num_coeffs = num_coeffs
        self.max_zernike_amplitude = max_zernike_amplitude

        self.num_coeffs = min(31, num_coeffs)

        self.kernel_generator = DiffractionBlurGenerator(psf_size=self.psf_size,
                                            max_zernike_amplitude=self.max_zernike_amplitude,
                                            zernike_index=range(2, self.num_coeffs+2),
                                            num_channels=1,
                                            **self.factorial_kwargs
                                            )
        
        self.objective_fn = LossFidelity(physics=conv2d_fft,
                                         **self.factorial_kwargs)
        
    
    def _coeffs_restarts(self, n_restarts, b, initialization_method):
        if initialization_method == 'sobol':
            coeffs_restarts = SobolEngine(dimension=self.num_coeffs
                                            ).draw(n_restarts * b
                                                    ).view(n_restarts, b,
                                                            self.num_coeffs
                                                            ).to(**self.factorial_kwargs)
            coeffs_restarts = self.max_zernike_amplitude * (coeffs_restarts - 0.5)
        else:
            if initialization_method != 'random':
                warn(f"Unknown initialization method: {initialization_method}. Using 'random' instead.")
            coeffs_restarts = 0.1 * torch.rand(n_restarts, b, self.num_coeffs,
                                                                      **self.factorial_kwargs)
        
        return coeffs_restarts
    
    
    def forward(self, img,
                  y_ref,
                  n_restarts=5,
                  niter=10,
                  initialization_method='random'):
        print("process")
        
        b = img.shape[0]
        best_coeffs = torch.zeros((b, self.num_coeffs), **self.factorial_kwargs)
        best_loss = None

        coeffs_restarts = self._coeffs_restarts(n_restarts, b, initialization_method)

        for coeffs in coeffs_restarts:
            loss = None
            coeffs = coeffs.clone().requires_grad_(True)    
            optimizer = LBFGS([coeffs],
                            lr=1.0,
                            history_size=10,
                            max_iter=niter,
                            line_search_fn='strong_wolfe')
            for _ in range(niter):
                def closure():
                    optimizer.zero_grad()
                    filter_est = self.generate_blur(coeffs)['filter']
                    loss = self.objective_fn(img, y_ref, filter_est, crop=False)
                    loss.backward()
                    return loss
                
                optimizer.step(closure)
                
            with torch.no_grad():
                loss = self.objective_fn(img, y_ref,
                                            self.generate_blur(coeffs)['filter'],
                                            crop=False).item()
            print(loss)
        
            if (best_loss is None) or (loss < best_loss):
                best_loss = loss
                best_coeffs = coeffs.detach()

        return best_coeffs
    
    def generate_blur(self, coeff):
        return self.kernel_generator.step(coeff=coeff)