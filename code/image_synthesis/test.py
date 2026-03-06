#%%
import os
project_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
print(project_abs_path)
import sys
sys.path.append(project_abs_path)
# %%
from PIL import Image
import torch
import torch.nn.functional as F
import deepinv as dinv
from putils import show_images
import numpy as np
from deepinv.physics import TiledSpaceVaryingBlur

# %% Load image
img = Image.open(os.path.join(project_abs_path, 'data/first_img.JPEG')).convert('L')
# resize to 256x256
img = img.resize((256, 256)) 
# convert to torch tensor
img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  
show_images(img_tensor)
# %% Generate diffraction blur kernels
from deepinv.physics.generator import DiffractionBlurGenerator
generator = DiffractionBlurGenerator((31, 31), num_channels=1)
print("\n".join(generator.zernike_polynomials)) # list of Zernike polynomials used
#%%
blur = generator.step(batch_size=49)  # dict_keys(['filter', 'coeff', 'pupil'])
filters = blur['filter'].transpose(0, 1).unsqueeze(0)
print(filters.shape)
# show_images([F.interpolate(blur['filter'], size=blur['pupil'].shape[1:]),
#              torch.real(blur['pupil'].unsqueeze(0))],
#             title=['filter', 'pupil'])
# %%
img_size = (256, 256)
patch_size = (64, 64)
stride = (32, 32)


physics = TiledSpaceVaryingBlur(patch_size=patch_size, stride=stride)
y = physics(img_tensor, filters=filters)
print(img_tensor.shape, y.shape)
dinv.utils.plot([img_tensor, y], titles=["Original", "Blurred"])

# %%
