import torch
import torch.nn.functional as F
from deepinv.physics import TiledSpaceVaryingBlur
from .utils import grad, as_pair
from torchmetrics.image import TotalVariation
from typing import Tuple, Union
from functools import partial
from warnings import warn
from .utils import psf_parameterization






class LossFidelity(torch.nn.Module):
    def __init__(self,
                 reduction: str='sum',
                 norm: str='l2',
                 physics: TiledSpaceVaryingBlur = None,
                 dtype=torch.float32,
                 device='cpu',
                 **kwargs):
        super().__init__(*kwargs)
        self.physics = physics
        if norm == 'l2':
            self.criterion = torch.nn.MSELoss(reduction=reduction).to(device=device, dtype=dtype)
        elif norm == 'l1':
            self.criterion = torch.nn.L1Loss(reduction=reduction).to(device=device, dtype=dtype)
        else:
            warn(f"Unsupported norm: {norm}, using L2 by default.")
            self.criterion = torch.nn.MSELoss(reduction=reduction).to(device=device, dtype=dtype)

    def forward(self,
               x: torch.Tensor,
               y: torch.Tensor, 
               filter: torch.Tensor = None,
               crop: bool=True):
        r'''
        Fidelity loss for blind deblurring
        Args:
            x: estimated image of shape (B, C, H, W)
            y: observed blurry image of shape (B, C, H, W)
            filters: space-varying blur kernels of shape (num_kernels, 1, kernel_size[0], kernel_size[1])
            physics: physics model to apply the blur kernels to the estimated image
        '''
        # assert x.shape == y.shape, "the original and blurred image must have the same shape"
        if filter is not None:
            x = self.physics(x, filter=filter)
        if crop:
            kh, kw = filter.shape[-2:]
            y = y[..., kh//2:-(kh//2), kw//2:-(kw//2) ]
        return self.criterion(x, y)
    
class RegImage(torch.nn.Module):
    def __init__(self,
                 device='cpu',
                 dtype=torch.float32,
                 reduction='sum',
                 **kwargs):
        super().__init__(*kwargs)
        
        self.criterion = TotalVariation(reduction=reduction).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        r'''
        Regularization loss for the estimated image
        Args:
            x: estimated image of shape (B, C, H, W)
        '''
        return self.criterion(x)
    
class RegFilter(torch.nn.Module):
    def __init__(self,
                 kernel_size: Union[Tuple[int, int], int],
                 num_kernels: Union[Tuple[int, int], int] = None,
                 reg_coeffs: Tuple[float, float, float] = (1.0, 0.0, 0.0),
                 reduction='sum',
                 r: int=3,
                 device: str='cpu',
                 dtype=torch.float32,
                 **kwargs):
        super().__init__(*kwargs)
        assert r >= 2, "r must be greater than or equal to 2"
        assert reduction in ['sum', 'mean'], "reduction must be either 'sum' or 'mean'"
        self.reduction = reduction
        self.num_kernels = as_pair(num_kernels)
        h, w = as_pair(kernel_size)
        hgrid, vgrid = torch.meshgrid(torch.arange(-(h//2), (h//2) + 1, device=device),
                                      torch.arange(-(w//2), (w//2) + 1, device=device),
                                      indexing='ij')
        self.mask = (1 + (hgrid/h)**2 + (vgrid/w)**2)**r
        self.reg_coeffs = reg_coeffs
        self.dtype = dtype
        

    def update_parameters(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def forward(self, h: torch.Tensor):
        r'''
        Regularization loss for the estimated filter
        Args:
            h: estimated filter of shape (num_kernels, 1, H, W)
        '''
        if self.num_kernels is None:
            raise ValueError("num_kernels must be set before calling forward")
        
        grad_h, grad_v = grad(h.view(*self.num_kernels, *h.shape[2:]), dim=(0,1))
        fft_h = torch.fft.fftshift(torch.fft.fft2(h, dim=(-2, -1)))

        if self.reduction == 'sum':
            R1 = 0 if self.reg_coeffs[0] == 0 else grad_h.abs().sum() + grad_v.abs().sum()
            R2 = 0 if self.reg_coeffs[1] == 0 else torch.sum((torch.sum(h, dim=(1,2,3)) - 1).pow(2))
            R3 = 0 if self.reg_coeffs[2] == 0 else torch.sum(self.mask[None, None, :, :]*(torch.abs(fft_h)**2))
        else:
            R1 = 0 if self.reg_coeffs[0] == 0 else torch.mean((torch.sum(h, dim=(1,2,3)) - 1).pow(2))
            R2 = 0 if self.reg_coeffs[1] == 0 else torch.mean(grad_h.abs()) + torch.mean(grad_v.abs())
            R3 = 0 if self.reg_coeffs[2] == 0 else torch.mean(self.mask[None, None, :, :]*(torch.abs(fft_h)**2))

        return self.reg_coeffs[0] * R1 + self.reg_coeffs[1] * R2 + self.reg_coeffs[2] * R3

class TotalLoss(torch.nn.Module):
    def __init__(self,
                 kernel_size: Union[Tuple[int, int], int],
                 num_kernels: Union[Tuple[int, int], int],
                 basis: str = "zernike",
                 filters_generator= None,
                 filters_reg_coeffs: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 r: int = 3,
                 device='cpu',
                 dtype=torch.float32,
                 reduction='sum',
                 physics: TiledSpaceVaryingBlur = None,
                 coeffs: Tuple[float, float] = (1.0, 1.0),
                 **kwargs):
        r'''
        Total loss for blind deblurring, including fidelity and regularization terms
        Args:
            kernel_size: size of the blur kernels, either a tuple (h, w) or an integer (h = w)
            num_kernels: number of blur kernels, either a tuple (num_kernels_y, num_kernels_x) or an integer (num_kernels_y = num_kernels_x)
            basis: type of basis used for the projection of the pupil function, either "zernike" or "pixel"
            filters_generator: generator of the filters if basis is "zernike", or a string in ["softmax", "relu", "silu"] for the parameterization of the filters if basis is "pixel"
            reg_coeffs: coefficients for the regularization terms on the filters, of the form (coeff_R1, coeff_R2, coeff_R3)
            r: exponent for the mask in the regularization term on the filters, must be greater than or equal to 2
            device: device to use for the computations
            dtype: data type to use for the computations
            reduction: reduction method for the losses, either "sum" or "mean"
            physics: physics model to apply the blur kernels to the estimated image, used for the fidelity term, need to be provided if basis is "zernike"
            coeffs: coefficients for the fidelity and regularization terms, of the form (coeff_reg_filter, coeff_reg_img)
        '''
        super().__init__(**kwargs)
        factory_kwargs = {'device': device,
                          'dtype': dtype,
                          'reduction': reduction}
        assert basis in ["zernike", "fourier"], "basis must be either 'zernike' or 'fourier'"
        assert basis == "zernike" or (filters_generator is not None), "filters_generator must be provided if basis is 'zernike'"

        self.basis = basis
        
        if basis == "pixel":
            self.filters_generator = partial(psf_parameterization,
                                             parameterization=filters_generator)
        else:
            self.filters_generator = filters_generator

        self.num_kernels = as_pair(num_kernels)

        self.criterion_fidel = LossFidelity(physics=physics,
                                            **factory_kwargs)
        
        if coeffs[0] > 0:
            self.criterion_reg_filter = RegFilter(kernel_size=kernel_size,
                                                num_kernels=num_kernels,
                                                reg_coeffs=filters_reg_coeffs,
                                                r=r,
                                                **factory_kwargs)
        else:
            self.criterion_reg_filter = lambda *arg,**kwargs: torch.tensor(0.0, device=device, dtype=dtype)

        if coeffs[1] > 0:        
            self.criterion_reg_img = RegImage(**factory_kwargs)
        else:
            self.criterion_reg_img = lambda *arg,**kwargs: torch.tensor(0.0, device=device, dtype=dtype)

        self.coeffs = coeffs

    def forward(self,
               x: torch.Tensor,
               y: torch.Tensor, 
               projection_coeffs: torch.Tensor):
        r'''
        Total loss for blind deblurring
        Args:
            x: estimated image of shape (B, C, H, W)
            y: observed blurry image of shape (B, C, H, W)
            projection_coeffs: coefficients of the projection of the pupil function onto a basis, of shape (num_kernels, num_atoms)
        '''
        if self.basis == "pixel":
            filters = torch.abs(projection_coeffs)**2
        elif self.basis == "zernike":
            filters = self.filters_generator.step(batch_size=self.num_kernels[0]*self.num_kernels[1],
                                                  coeff=projection_coeffs)['filter']
            
        loss_fidel = self.criterion_fidel(x, y, filters=filters.transpose(0, 1).unsqueeze(0))
        loss_reg_filter = self.criterion_reg_filter(filters) if filters is not None else 0
        loss_reg_img = self.criterion_reg_img(x)
        return torch.stack([
            loss_fidel,
            self.coeffs[0] * loss_reg_filter,
            self.coeffs[1] * loss_reg_img
        ])