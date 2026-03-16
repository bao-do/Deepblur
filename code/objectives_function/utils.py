import torch
from torch import fft
from torch.nn.functional import softmax, relu, silu
from warnings import warn

def grad(input, dim=3):

    if isinstance(dim, (list, tuple)):
        return [grad(input, d) for d in dim]

    gindex = [slice(None)] * input.dim()
    gindex[dim] = slice(1, None)

    lindex = [slice(None)] * input.dim()
    lindex[dim] = slice(None, -1)
    return input[tuple(gindex)] - input[tuple(lindex)]

def as_pair(x):
    if isinstance(x, (list, tuple)):
        return x
    else:
        return (x, x)
    
def blur_fn_invariant(x, filters):
    kh, kw = filters.shape[-2:]
    kernel_padded = torch.zeros(1,1, *x.shape[-2:], device=x.device, dtype=x.dtype)
    kernel_padded[:, :, :kh, :kw] = filters
    kernel_padded = torch.roll(kernel_padded, shifts=(-(kh//2), -(kw//2)), dims=(-2, -1))
    x_fft = fft.fft2(x)
    kernel_fft = fft.fft2(kernel_padded)
    y_fft = x_fft * kernel_fft
    y = torch.real(fft.ifft2(y_fft)[...,kh//2:-(kh//2), kw//2:-(kw//2)])
    return y

def psf_parameterization(coeffs, parameterization="softmax"):
    if parameterization == "softmax":
        kernel_est = softmax(coeffs.flatten(), dim=-1).view(coeffs.shape)
    elif parameterization == "relu":
        kernel_est = relu(coeffs)
        kernel_est = kernel_est / kernel_est.sum()
    elif parameterization == "silu":
        kernel_est = silu(coeffs)
        kernel_est = kernel_est / kernel_est.sum()
    else:
        warn(f"Unsupported parameterization: {parameterization}, using softmax by default.")
        kernel_est = softmax(coeffs.flatten(), dim=-1).view(coeffs.shape)
    return kernel_est