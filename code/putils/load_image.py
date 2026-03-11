from PIL import Image
import torch
from typing import Tuple, List
import numpy as np

def open_image(image_path: str,
             img_size: Tuple[int, int]=None,
             gray_scale: bool=True,
             device='cpu',
             dtype=torch.float32):
    '''Open an image and convert it to a torch tensor of shape (1, 1, H, W)
    and normalized to [0, 1].'''
    img = Image.open(image_path)
    print(f"Opened image with size {img.size} and mode {img.mode}")
    if gray_scale and (not img.mode.startswith(('L','I'))):
        img = img.convert('L')
    if img_size:
        img = img.resize(img_size) 
    img_tensor = torch.from_numpy(np.array(img)).float()
    img_tensor = img_tensor / img_tensor.max()
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)
    return img_tensor