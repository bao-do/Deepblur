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
                                         reduction='sum',
                                         **self.factorial_kwargs)
        
    
    def _coeffs_restarts(self, n_restarts, initialization_method):
        if initialization_method == 'sobol':
            coeffs_restarts = SobolEngine(dimension=self.num_coeffs,
                                scramble=True,
                                seed=random_seed()).draw(n_restarts)
            
            coeffs_restarts = self.max_zernike_amplitude * (coeffs_restarts - 0.5)
            
        else:
            if initialization_method != 'random':
                warn(f"Unknown initialization method: {initialization_method}. Using 'random' instead.")
            coeffs_restarts = 0.1 * torch.rand(n_restarts, self.num_coeffs)
        
        return coeffs_restarts
    
    
    def forward(self, img,
                  y_ref,
                  n_restarts=5,
                  niter=10,
                  initialization_method='random'):
        
        assert initialization_method in ['random', 'sobol'], "Initialization method must be either 'random' or 'sobol'."
        assert (img.shape[0] == 1) and (y_ref.shape[0] == 1), "Batch size of img and y_ref must be the same."
        
        best_coeffs = torch.zeros((1, self.num_coeffs), **self.factorial_kwargs)
        best_loss = None

        coeffs_restarts = self._coeffs_restarts(n_restarts, initialization_method)

        for coeffs in coeffs_restarts:
            coeffs = coeffs.unsqueeze(0).clone().to(**self.factorial_kwargs).requires_grad_(True)    # Shape: (1, num_coeffs)
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
                                            crop=False)
        
            if (best_loss is None) or (loss < best_loss):
                best_loss = loss
                best_coeffs = coeffs.detach()

        return best_coeffs
    
    def generate_blur(self, coeff):
        return self.kernel_generator.step(coeff=coeff)
    


import torch
import torch.nn as nn

import numpy as np 
from deepinv.physics.functional import conv2d
import deepinv as dinv
import deepinv.physics.generator


# A set of functions to given (y,x) and the relationship y = h*x + e, retrieve h

def aberrations_to_zernike_coeffs(aberrations, scaling_factor=1.0):
    """
    Convert aberrations dict to Zernike coefficients for DiffractionBlurGenerator.

    Args:
        aberrations: dict like {'focus': 200, 'asti0': 50, 'coma0': -40}
        scaling_factor: conversion factor from nm to the scale expected by generator
                       (will be determined later based on NA, wavelength, etc.)

    Returns:
        tuple of (zernike_index, coefficients) where:
        - zernike_index: tuple of Zernike indices in Noll convention
        - coefficients: np.array of corresponding coefficient values
    """
    # Mapping from your aberration names to Noll indices
    # Based on Lakshminarayanan & Fleck (2011) and standard Noll indexing
    # Reference: Table 1 in the paper shows (n,m) to classical aberration mapping
    aberration_to_noll = {
        # Low order (n=2): j=3,4,5
        "focus": 4,  # Defocus (n=2, m=0), Noll j=4
        "asti0": 6,  # Vertical Astigmatism (n=2, m=2), Noll j=6
        "asti45": 5,  # Oblique Astigmatism (n=2, m=-2), Noll j=5
        # Third order (n=3): j=6,7,8,9
        "coma0": 7,  # Vertical Coma (n=3, m=-1), Noll j=7
        "coma90": 8,  # Horizontal Coma (n=3, m=1), Noll j=8
        "trefoil0": 9,  # Vertical Trefoil (n=3, m=-3), Noll j=9
        "trefoil30": 10,  # Oblique Trefoil (n=3, m=3), Noll j=10
        # Fourth order (n=4): j=11-14
        "spherical3": 11,  # Primary Spherical (n=4, m=0), Noll j=11
        "tetrafoil0": 14,  # (See convention of Sylvain's doc) 
        "asti5_0": 12,  # Secondary Vertical Astigmatism (n=5, m=-1) - check your data
        "coma5_0": 17,  # Secondary Coma (n=5, m=-1), Noll j=15 - check your data
        # Sixth order (n=6): j=21-27
        "spherical5": 22,  # Secondary Spherical (n=6, m=0), Noll j=22
        # Note: 'asti5_0' and 'coma5_0' naming is ambiguous - need to verify
        # what "5th" refers to in your nomenclature (radial order or something else)
    }

    # Collect the indices and coefficients
    zernike_indices = []
    coefficients = []

    for aber_name, value in aberrations.items():
        if (
            aber_name in aberration_to_noll
            and aberration_to_noll[aber_name] is not None
        ):
            noll_idx = aberration_to_noll[aber_name]
            zernike_indices.append(noll_idx)
            # Apply scaling: convert from nm to generator units
            coefficients.append(value * scaling_factor)

    if not zernike_indices:
        return tuple(), np.array([])

    return tuple(zernike_indices), np.array(coefficients)


def create_psf_from_aberrations(
    aberrations, psf_size=(51, 51), scaling_factor=1.0, fc=0.2, device="cpu"
):
    """
    Create a PSF using DiffractionBlurGenerator from aberration dict.

    Args:
        aberrations: dict like {'focus': 200, 'asti0': 50}
        psf_size: size of PSF to generate
        scaling_factor: conversion from nm to generator units
        fc: cutoff frequency
        device: torch device

    Returns:
        dict with:
            - 'filter': PSF as torch tensor of shape (1, 1, H, W)
            - 'zernike_indices': tuple of Zernike indices used
            - 'coefficients': the coefficient values used
    """
    import torch
    import deepinv

    # Convert aberrations to Zernike format
    zernike_indices, coeffs = aberrations_to_zernike_coeffs(aberrations, scaling_factor)

    if len(zernike_indices) == 0:
        # No aberrations - return ideal diffraction-limited PSF (all coeffs = 0)
        print("  No aberrations detected, generating ideal PSF")
        # Use default indices but with zero coefficients
        default_indices = (
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
        )  # Default from DiffractionBlurGenerator
        generator = deepinv.physics.generator.DiffractionBlurGenerator(
            psf_size=psf_size, zernike_index=default_indices, fc=fc, device=device
        )
        # Generate with zero coefficients
        coeff_tensor = torch.zeros(
            1, len(default_indices), device=device, dtype=torch.float32
        )
        result = generator.step(batch_size=1, coeff=coeff_tensor)

        return {
            "filter": result["filter"],
            "zernike_indices": default_indices,
            "coefficients": coeff_tensor.cpu().numpy(),
        }
    else:
        # Create generator with the SPECIFIC Zernike indices from your data
        print(f"  Using Zernike indices: {zernike_indices}")
        print(f"  With coefficients (scaled): {coeffs}")

        generator = deepinv.physics.generator.DiffractionBlurGenerator(
            psf_size=psf_size,
            zernike_index=zernike_indices,  # Only the indices we need!
            fc=fc,
            max_zernike_amplitude=0.15,  # Not used when we provide coeff explicitly
            device=device,
        )

        # Convert coefficients to torch tensor (shape: batch_size x num_coefficients)
        coeff_tensor = torch.tensor(
            coeffs, dtype=torch.float32, device=device
        ).unsqueeze(0)

        print(f"  Coefficient tensor shape: {coeff_tensor.shape}")

        # Generate PSF with these EXACT coefficients (not random!)
        result = generator.step(batch_size=1, coeff=coeff_tensor)

        return {
            "filter": result["filter"],  # Shape: (1, 1, H, W)
            "zernike_indices": zernike_indices,
            "coefficients": coeff_tensor.cpu().numpy(),
        }



@torch.no_grad()
def projection_simplex_sort(v: torch.Tensor) -> torch.Tensor:
    r"""
    Projects a tensor onto the simplex using a sorting algorithm.
    """
    shape = v.shape
    B = shape[0]
    v = v.view(B, -1)
    n_features = v.size(1)
    u = torch.sort(v, descending=True, dim=-1).values
    cssv = torch.cumsum(u, dim=-1) - 1.0
    ind = torch.arange(n_features, device=v.device)[None, :].expand(B, -1) + 1.0
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = torch.maximum(v - theta, torch.zeros_like(v))
    return w.reshape(shape)


def estimate_psf_zernike(
    reference,
    aberrated,
    psf_size=51,
    zernike_indices=None,
    fc=0.2,
    num_iterations=1000,
    lr=0.01,
    momentum=0.9,
    device="cpu",
    verbose=True,
):
    """
    Estimate PSF via Zernike polynomial coefficients.

    The PSF is generated using DiffractionBlurGenerator with Zernike decomposition.
    This is physically meaningful and includes optical aberrations.

    Solves: min_coeffs ||conv(reference, PSF(coeffs)) - aberrated||^2
    where PSF(coeffs) is generated via Zernike polynomials.

    Args:
        reference: reference image, numpy array or torch tensor (H, W)
        aberrated: aberrated image, numpy array or torch tensor (H, W)
                   Must be same size as reference
        psf_size: size of PSF
        zernike_indices: tuple of Noll indices to optimize.
                        If None, uses tuple(range(2, 29))
                        which includes tip/tilt for shift + common aberrations
        fc: cutoff frequency for diffraction (default 0.2)
        num_iterations: number of optimization iterations
        lr: learning rate for coefficients
        momentum: momentum coefficient
        device: 'cpu' or 'cuda'
        verbose: print progress

    Returns:
        dict with keys:
            'coefficients': optimized Zernike coefficients
            'zernike_indices': the Noll indices used
            'psf': final PSF
            'loss_history': loss over iterations
            'alpha': estimated intensity scale factor
    """

    # ========== 1. Setup ==========
    if isinstance(reference, np.ndarray):
        reference = torch.from_numpy(reference).float()
    if isinstance(aberrated, np.ndarray):
        aberrated = torch.from_numpy(aberrated).float()

    reference = reference.to(device)
    aberrated = aberrated.to(device)

    # Normalize reference to [0,1]
    ref_min, ref_max = reference.min(), reference.max()
    reference = (reference - ref_min) / (ref_max - ref_min + 1e-8)

    # Add dimensions: (H, W) -> (1, 1, H, W)
    if reference.ndim == 2:
        reference = reference.unsqueeze(0).unsqueeze(0)
    if aberrated.ndim == 2:
        aberrated = aberrated.unsqueeze(0).unsqueeze(0)

    # ========== 2. Zernike indices ==========
    if zernike_indices is None:
        zernike_indices = tuple(range(2, 29))

    num_coeffs = len(zernike_indices)

    if verbose:
        print(f"Zernike indices: {zernike_indices}")
        print(f"Number of coefficients: {num_coeffs}")

    # ========== 3. Initialize generator ==========
    generator = deepinv.physics.generator.DiffractionBlurGenerator(
        psf_size=psf_size, zernike_index=zernike_indices, fc=fc, device=device
    )

    # ========== 4. Initialize coefficients ==========
    coeffs = torch.zeros(num_coeffs, device=device, requires_grad=True)

    # Momentum buffer
    velocity = torch.zeros_like(coeffs) if momentum > 0 else None

    # ========== 5. Optimization loop ==========
    loss_history = []

    if verbose:
        print(
            f"\nZernike PSF Estimation: {psf_size}x{psf_size}, {num_iterations} iterations"
        )
        print(f"Learning rate: {lr}, Momentum: {momentum}\n")

    for iteration in range(num_iterations):
        # Zero gradients
        if coeffs.grad is not None:
            coeffs.grad.zero_()

        # Generate PSF from current coefficients
        coeff_batch = coeffs.unsqueeze(0)  # (num_coeffs,) -> (1, num_coeffs)
        psf_dict = generator.step(batch_size=1, coeff=coeff_batch)
        psf = psf_dict["filter"]  # Shape: (1, 1, psf_size, psf_size)

        # Forward: convolve reference with PSF
        reconstructed = conv2d(reference, psf, padding="valid")

        # Compute optimal scale: alpha = <recon, aber> / <recon, recon>
        alpha = (reconstructed * aberrated).sum() / (
            reconstructed * reconstructed
        ).sum().clamp(min=1e-10)

        # Loss: MSE with optimal scale
        loss = ((alpha * reconstructed - aberrated) ** 2).mean()
        loss_history.append(loss.item())

        # Backward
        loss.backward()

        # Update coefficients with momentum
        with torch.no_grad():
            if momentum > 0:
                velocity.mul_(momentum).add_(coeffs.grad)
                coeffs.sub_(lr * velocity)
            else:
                coeffs.sub_(lr * coeffs.grad)

        # Progress
        if verbose and (iteration % 100 == 0 or iteration == num_iterations - 1):
            # dinv.utils.plot(psf)
            print(
                f"Iter {iteration:4d}: Loss={loss.item():.6f}, Alpha={alpha.item():.4f}"
            )
            if iteration % 10 == 0:
                # Show top 3 coefficients
                top_idx = torch.topk(coeffs.abs(), min(3, num_coeffs)).indices
                coeff_str = ", ".join(
                    [f"Z{zernike_indices[i]}={coeffs[i].item():.3f}" for i in top_idx]
                )
                print(f"         Top coeffs: {coeff_str}")

    # ========== 6. Final results ==========
    with torch.no_grad():
        # Generate final PSF
        final_psf_dict = generator.step(batch_size=1, coeff=coeffs.unsqueeze(0))
        final_psf = final_psf_dict["filter"]

        # Final reconstruction
        final_recon = conv2d(reference, final_psf, padding="valid")
        final_alpha = (final_recon * aberrated).sum() / (
            final_recon * final_recon
        ).sum().clamp(min=1e-10)

    if verbose:
        dinv.utils.plot(
            {"aber:": aberrated, "final_recon": final_recon, "ref": reference}
        )
        print(f"\nFinal: Alpha={final_alpha.item():.4f}, Loss={loss_history[-1]:.6e}")

    return {
        "coefficients": coeffs.detach().cpu().numpy(),
        "zernike_indices": zernike_indices,
        "psf": final_psf.squeeze().cpu(),
        "loss_history": loss_history,
        "alpha": final_alpha.item(),
        "reconstructed": (final_alpha * final_recon).squeeze().cpu(),
        "reference": reference.squeeze().cpu(),
        "aberrated": aberrated.squeeze().cpu(),
    }


def estimate_psf_nonnegative(
    reference,
    aberrated,
    psf_size=51,
    num_iterations=500,
    lr=None,
    momentum=0.95,
    device="cpu",
    verbose=True,
):
    """
    Estimate PSF by deconvolution with non-negativity constraint.

    Solves: min_h ||conv(reference, h) - aberrated||^2
    Subject to: h >= 0, sum(h) = 1

    The loss is scale-invariant: finds optimal alpha such that
    ||alpha * conv(reference, h) - aberrated||^2 is minimized.

    Args:
        reference: reference (no aberration) image, numpy array or torch tensor (H, W)
        aberrated: aberrated image, numpy array or torch tensor (H, W)
        psf_size: size of PSF to estimate (square: psf_size x psf_size)
        num_iterations: number of optimization iterations
        lr: learning rate. If None, uses 1/||reference||_1 (optimal for convex quadratic)
        momentum: momentum coefficient (0.95 default, 0 to disable)
        device: 'cpu' or 'cuda'
        verbose: print progress

    Returns:
        dict with keys:
            'psf': estimated PSF (psf_size, psf_size)
            'loss_history': loss values over iterations
            'alpha': estimated intensity scale factor
            'reconstructed': final reconstruction (alpha * conv(ref, psf))
    """

    # ========== 1. Setup ==========
    # Convert to torch and add batch/channel dims
    if isinstance(reference, np.ndarray):
        reference = torch.from_numpy(reference).float()
    if isinstance(aberrated, np.ndarray):
        aberrated = torch.from_numpy(aberrated).float()

    reference = reference.to(device)
    aberrated = aberrated.to(device)

    # Normalize reference to [0,1] for numerical stability
    ref_min, ref_max = reference.min(), reference.max()
    reference = (reference - ref_min) / (ref_max - ref_min + 1e-8)

    # Add dimensions: (H, W) -> (1, 1, H, W)
    if reference.ndim == 2:
        reference = reference.unsqueeze(0).unsqueeze(0)
    if aberrated.ndim == 2:
        aberrated = aberrated.unsqueeze(0).unsqueeze(0)

    # ========== 2. Learning rate ==========
    if lr is None:
        # Lipschitz constant L = ||reference||_1 for non-negative images
        L = reference.sum().item() / 1000
        lr = 1.0 / L
        if verbose:
            print(f"Auto LR: L={L:.2e}, lr=1/L={lr:.2e}")

    # ========== 3. Initialize PSF ==========
    psf = torch.zeros(1, 1, psf_size, psf_size, device=device, requires_grad=True)
    center = psf_size // 2
    y, x = torch.meshgrid(
        torch.arange(psf_size, device=device),
        torch.arange(psf_size, device=device),
        indexing="ij",
    )

    # Start with Gaussian centered at PSF center
    with torch.no_grad():
        psf.data[0, 0] = torch.exp(
            -((x - center) ** 2 + (y - center) ** 2) / (2 * (psf_size / 6) ** 2)
        )
        psf.data /= psf.data.sum()  # Normalize

    # Momentum buffer
    velocity = torch.zeros_like(psf.data) if momentum > 0 else None

    # ========== 4. Optimization loop ==========
    loss_history = []

    if verbose:
        print(f"\nPSF Estimation: {psf_size}x{psf_size}, {num_iterations} iterations")
        print(f"Momentum: {momentum}\n")

    for iteration in range(num_iterations):
        # Zero gradients
        if psf.grad is not None:
            psf.grad.zero_()

        # Forward: convolve reference with PSF
        reconstructed = conv2d(reference, psf, padding="valid")

        # Compute optimal scale: alpha = <recon, aber> / <recon, recon>
        alpha = (reconstructed * aberrated).sum() / (
            reconstructed * reconstructed
        ).sum().clamp(min=1e-10)

        # Loss: MSE with optimal scale
        loss = ((alpha * reconstructed - aberrated) ** 2).mean()
        loss_history.append(loss.item())

        # Backward
        loss.backward()

        # Update with momentum
        with torch.no_grad():
            if momentum > 0:
                velocity.mul_(momentum).add_(psf.grad)
                psf.data.sub_(lr * velocity)
            else:
                psf.data.sub_(lr * psf.grad)

            # Project: non-negativity and sum=1
            psf.data = projection_simplex_sort(psf.data)

        # Progress
        if verbose and (iteration % 50 == 0 or iteration == num_iterations - 1):
            # dinv.utils.plot(psf.data)
            print(
                f"Iter {iteration:4d}: Loss={loss.item():.6f}, Alpha={alpha.item():.4f}"
            )

    # ========== 5. Final reconstruction ==========
    with torch.no_grad():
        final_recon = conv2d(reference, psf, padding="valid")

        final_alpha = (final_recon * aberrated).sum() / (
            final_recon * final_recon
        ).sum().clamp(min=1e-10)
        final_recon_scaled = final_alpha * final_recon

    if verbose:
        dinv.utils.plot(
            {"aber:": aberrated, "final_recon": final_recon, "ref": reference}
        )
        print(f"\nFinal: Alpha={final_alpha.item():.4f}, Loss={loss_history[-1]:.6e}\n")

    # Return everything as CPU numpy/tensors
    return {
        "psf": psf.squeeze().detach().cpu(),
        "loss_history": loss_history,
        "alpha": final_alpha.item(),
        "reconstructed": final_recon_scaled.squeeze().cpu(),
    }


# ---------------------------------------------------------------------------
# Small MLP:  R^input_dim  ->  R^num_coeffs
# ---------------------------------------------------------------------------
class CoefficientMLP(nn.Module):
    """
    Tiny MLP used as the overparameterized reparameterisation.
 
    Architecture: Linear -> Tanh -> Linear -> Tanh -> Linear
    The output is *unbounded* (no final activation) so Zernike coefficients
    can take any real value.
 
    Args:
        input_dim : dimension P of the fixed noise seed  (P > num_coeffs)
        num_coeffs: output dimension = number of Zernike coefficients
        hidden_dim: width of hidden layers
    """
 
    def __init__(self, input_dim: int, num_coeffs: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_coeffs),
        )
        # Small-weight init so we start near zero coefficients
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
 
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
 
 
# ---------------------------------------------------------------------------
# Main estimation function
# ---------------------------------------------------------------------------
def estimate_psf_zernike_mlp(
    reference,
    aberrated,
    psf_size: int = 51,
    zernike_indices=None,
    fc: float = 0.2,
    # --- L-BFGS knobs ---
    num_iterations: int = 200,
    lbfgs_history: int = 20,
    lbfgs_max_iter: int = 4,          # inner CG steps per outer step
    lr: float = 0.5,
    # --- MLP overparameterization ---
    use_mlp: bool = True,
    mlp_input_dim: int = 256,         # P  (should be > num_coeffs)
    mlp_hidden_dim: int = 128,
    # --- misc ---
    device: str = "cpu",
    verbose: bool = True,
):
    """
    Estimate PSF via Zernike polynomial coefficients.
 
    The PSF is generated using DiffractionBlurGenerator with Zernike decomposition.
 
    Solves:  min  ||alpha * conv(reference, PSF(coeffs)) - aberrated||^2
 
    Two improvements over vanilla gradient descent
    -----------------------------------------------
    1. **L-BFGS** optimizer  — uses curvature information for much faster
       convergence on smooth objectives.
 
    2. **MLP overparameterization** (use_mlp=True, Deep-Image-Prior spirit):
       Instead of optimising `coeffs` directly we optimise the weights θ of a
       small MLP  f_θ : R^P → R^num_coeffs  and set
           coeffs = f_θ(z)
       where z ∈ R^P is a *fixed* random seed.  The implicit bias of the
       network towards smooth solutions acts as a regulariser and helps
       escape bad local minima.
 
    Args:
        reference    : reference image, numpy array or torch tensor (H, W)
        aberrated    : aberrated image, numpy array or torch tensor (H, W)
        psf_size     : side length of the square PSF kernel
        zernike_indices: Noll indices to optimise.  Defaults to range(2, 29).
        fc           : diffraction cutoff frequency (default 0.2)
        num_iterations: number of L-BFGS outer steps
        lbfgs_history : L-BFGS history size (line-search memory)
        lbfgs_max_iter: L-BFGS inner iterations per outer step
        lr           : learning rate passed to L-BFGS (acts as step-size bound)
        use_mlp      : if True, optimise MLP weights instead of coeffs directly
        mlp_input_dim: dimension P of the fixed MLP input seed  (P > num_coeffs)
        mlp_hidden_dim: hidden layer width of the MLP
        device       : 'cpu' or 'cuda'
        verbose      : print progress
 
    Returns:
        dict with keys:
            'coefficients'  : optimised Zernike coefficients  (numpy, num_coeffs)
            'zernike_indices': Noll indices used
            'psf'           : final PSF tensor  (psf_size, psf_size)
            'loss_history'  : loss value at each outer L-BFGS step
            'alpha'         : estimated intensity scale factor
            'reconstructed' : alpha * conv(reference, PSF)
            'reference'     : normalised reference (squeezed)
            'aberrated'     : aberrated image (squeezed)
    """
 
    # ------------------------------------------------------------------ #
    # 1. Tensor setup                                                      #
    # ------------------------------------------------------------------ #
    if isinstance(reference, np.ndarray):
        reference = torch.from_numpy(reference).float()
    if isinstance(aberrated, np.ndarray):
        aberrated = torch.from_numpy(aberrated).float()
 
    reference = reference.to(device)
    aberrated = aberrated.to(device)
 
    # Normalise reference to [0, 1]
    ref_min, ref_max = reference.min(), reference.max()
    reference = (reference - ref_min) / (ref_max - ref_min + 1e-8)
 
    # (H, W) -> (1, 1, H, W)
    if reference.ndim == 2:
        reference = reference.unsqueeze(0).unsqueeze(0)
    if aberrated.ndim == 2:
        aberrated = aberrated.unsqueeze(0).unsqueeze(0)
 
    # ------------------------------------------------------------------ #
    # 2. Zernike indices                                                   #
    # ------------------------------------------------------------------ #
    if zernike_indices is None:
        zernike_indices = tuple(range(2, 29))
    num_coeffs = len(zernike_indices)
 
    if verbose:
        print(f"Zernike indices : {zernike_indices}")
        print(f"Num coefficients: {num_coeffs}")
 
    # ------------------------------------------------------------------ #
    # 3. Diffraction PSF generator                                         #
    # ------------------------------------------------------------------ #
    generator = dinv.physics.generator.DiffractionBlurGenerator(
        psf_size=psf_size, zernike_index=zernike_indices, fc=fc, device=device
    )
 
    # ------------------------------------------------------------------ #
    # 4. Parameterisation: direct coeffs  OR  MLP overparameterization    #
    # ------------------------------------------------------------------ #
    if use_mlp:
        if mlp_input_dim <= num_coeffs:
            raise ValueError(
                f"mlp_input_dim ({mlp_input_dim}) must be > num_coeffs ({num_coeffs}) "
                "for overparameterization to make sense."
            )
        # Fixed random seed — never updated
        z = torch.randn(1, mlp_input_dim, device=device)
        z.requires_grad_(False)
 
        mlp = CoefficientMLP(mlp_input_dim, num_coeffs, mlp_hidden_dim).to(device)
        params = list(mlp.parameters())
 
        def get_coeffs() -> torch.Tensor:
            """Forward pass through MLP to obtain Zernike coefficients."""
            return mlp(z).squeeze(0)          # (num_coeffs,)
 
        if verbose:
            n_params = sum(p.numel() for p in params)
            print(f"\nMLP overparameterization: {mlp_input_dim} -> "
                  f"{mlp_hidden_dim} -> {mlp_hidden_dim} -> {num_coeffs}")
            print(f"Total MLP parameters: {n_params}  (vs {num_coeffs} direct coeffs)")
 
    else:
        # Direct optimisation of Zernike coefficients
        coeffs_direct = torch.zeros(num_coeffs, device=device, requires_grad=True)
        params = [coeffs_direct]
 
        def get_coeffs() -> torch.Tensor:
            return coeffs_direct
 
        if verbose:
            print("\nDirect coefficient optimisation (no MLP).")
 
    # ------------------------------------------------------------------ #
    # 5. L-BFGS optimizer                                                  #
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.LBFGS(
        params,
        lr=lr,
        max_iter=lbfgs_max_iter,
        history_size=lbfgs_history,
        line_search_fn="strong_wolfe",   # crucial for robust convergence
    )
 
    loss_history: list[float] = []
 
    if verbose:
        print(f"\nL-BFGS PSF estimation: psf={psf_size}x{psf_size}, "
              f"outer_iters={num_iterations}, inner_iter={lbfgs_max_iter}")
        print(f"lr={lr}, history={lbfgs_history}\n")
 
    # ------------------------------------------------------------------ #
    # 6. Optimisation loop                                                 #
    # ------------------------------------------------------------------ #
    # We keep a mutable container so the closure can write alpha out
    state = {"alpha": 1.0, "loss": float("inf")}
 
    def closure() -> torch.Tensor:
        optimizer.zero_grad()
 
        coeffs = get_coeffs()                           # (num_coeffs,)
        coeff_batch = coeffs.unsqueeze(0)               # (1, num_coeffs)
 
        psf_dict = generator.step(batch_size=1, coeff=coeff_batch)
        psf = psf_dict["filter"]                        # (1, 1, H, W)
 
        reconstructed = conv2d(reference, psf, padding="valid")
 
        # Optimal closed-form scale
        alpha = (reconstructed * aberrated).sum() / (
            reconstructed * reconstructed
        ).sum().clamp(min=1e-10)
 
        #loss = ((alpha * reconstructed - aberrated) ** 2).mean()
        loss = ((alpha * reconstructed - aberrated).abs()).mean()
        loss.backward()
 
        state["alpha"] = alpha.item()
        state["loss"] = loss.item()
        return loss
 
    for iteration in range(num_iterations):
        optimizer.step(closure)
        loss_history.append(state["loss"])
 
        if verbose and (iteration % 20 == 0 or iteration == num_iterations - 1):
            coeffs_now = get_coeffs().detach()
            top_idx = torch.topk(coeffs_now.abs(), min(3, num_coeffs)).indices
            coeff_str = ", ".join(
                [f"Z{zernike_indices[i]}={coeffs_now[i].item():.3f}" for i in top_idx]
            )
            print(f"Iter {iteration:4d}: Loss={state['loss']:.6e}, "
                  f"Alpha={state['alpha']:.4f} | top coeffs: {coeff_str}")
 
    # ------------------------------------------------------------------ #
    # 7. Final results                                                     #
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        final_coeffs = get_coeffs()
        tilt_mask = torch.tensor([i in {1, 2} for i in zernike_indices], device=device)
        final_coeffs = final_coeffs * (~tilt_mask) # Discarding the psf shift
        final_psf_dict = generator.step(
            batch_size=1, coeff=final_coeffs.unsqueeze(0)
        )
        final_psf = final_psf_dict["filter"]
 
        final_recon = conv2d(reference, final_psf, padding="valid")
        final_alpha = (final_recon * aberrated).sum() / (
            final_recon * final_recon
        ).sum().clamp(min=1e-10)
 
    if verbose:
        dinv.utils.plot(
            {"aberrated": aberrated, "final_recon": final_alpha * final_recon, "reference": reference, "est psf": final_psf}
        )
        print(f"\nFinal: Alpha={final_alpha.item():.4f}, Loss={loss_history[-1]:.6e}")
 
    return {
        "coefficients": final_coeffs.detach().cpu().numpy(),
        "zernike_indices": zernike_indices,
        "psf": final_psf.squeeze().cpu(),
        "loss_history": loss_history,
        "alpha": final_alpha.item(),
        "reconstructed": (final_alpha * final_recon).squeeze().cpu(),
        "reference": reference.squeeze().cpu(),
        "aberrated": aberrated.squeeze().cpu(),
    }