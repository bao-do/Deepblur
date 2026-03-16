import torch 
from torch import fft


def initial_coeffs_invariant(psf_size, x, y):
    fft_x = fft.fft2(x)
    fft_y = fft.fft2(y)
    fft_h = fft_y / (fft_x + 1e-8)
    h = torch.real(fft.ifft2(fft_h))
    h = torch.roll(h, shifts=(psf_size[0]//2, psf_size[1]//2), dims=(-2, -1))
    h_cropped = h[..., :psf_size[0], :psf_size[1]]
    return h_cropped