#%%
import torch
from deepinv.physics import TiledSpaceVaryingBlur
from deepinv.physics.generator import MotionBlurGenerator, TiledBlurGenerator
import deepinv as dinv
img_size = (256, 256)
patch_size = (64, 64)
stride = (32, 32)
x = dinv.utils.load_example(
       "butterfly.png", img_size=img_size, resize_mode="resize"
   )
psf_generator = MotionBlurGenerator(psf_size=(31, 31))
generator = TiledBlurGenerator(
    psf_generator=psf_generator,
    patch_size=patch_size,
    stride=stride,
)
filters = generator.step(batch_size=1, img_size=img_size)["filters"]
physics = TiledSpaceVaryingBlur(patch_size=patch_size, stride=stride)
y = physics(x, filters=filters)
print(x.shape, y.shape)
dinv.utils.plot([x, y], titles=["Original", "Blurred"])
# %%
