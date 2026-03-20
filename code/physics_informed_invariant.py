#%%
import torch
import matplotlib.pyplot as plt
from putils import open_image, show_images
import os
import torch.fft as fft
import deepinv as dinv
from torchvision.utils import save_image
from deepinv.physics.generator import DiffractionBlurGenerator
from torch.optim import  LBFGS
from objectives_function import LossFidelity, blur_fn_invariant
from deepinv.physics.functional import conv2d_fft

absolute_path = os.path.abspath(os.path.dirname(__file__))
figure_path = os.path.abspath(os.path.join(absolute_path, ".."))
figure_path = os.path.join(figure_path, "tex/figures/physics_informed_invariant")

exp_type = 'simulation'
figure_path = os.path.join(figure_path, exp_type)

os.makedirs(figure_path, exist_ok=True)

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
kwargs = {'device': device, 'dtype': dtype}
img_size = (256, 256)
psf_size = (31, 31)
num_coeffs = 15

img = open_image(os.path.join(absolute_path, "data/first_img.JPEG"), 
                 img_size=img_size,
                 **kwargs)
img_crop = img[..., psf_size[0]//2:-(psf_size[0]//2), psf_size[1]//2:-(psf_size[1]//2)]
show_images([img], title=["Original Image"])
save_image(img, os.path.join(figure_path, "original_image.png"))
# %%
sigma_list = torch.tensor([0, 0.01, 0.05, 0.1], **kwargs)
# sigma_list = torch.tensor([0], **kwargs)
sample_per_sigma = 50

max_zernike_amplitude = 0.3

kernel_generator = DiffractionBlurGenerator(psf_size=psf_size,
                                            max_zernike_amplitude=max_zernike_amplitude,
                                            zernike_index=range(2, 12),
                                            num_channels=1,
                                            **kwargs)
def random_seed():
    return torch.randint(0, 10000, (1,)).item()

blurs = kernel_generator.step(batch_size=sample_per_sigma, seed=random_seed())
kernels = blurs['filter']
pupils = blurs['pupil']

y = conv2d_fft(img.expand(sample_per_sigma, -1, -1, -1), kernels, padding="valid")

# y = blur_fn_invariant(img, kernels)

# show_images([torch.real(pupils)], title=["Pupil Function"])
# show_images([kernels], title=["Original Kernel"])
# show_images([y], title=["Blurred Image"])




#%%
niter = 30
learning_rate = 1e-3
eta_min = 1e-8

n_restarts = 5

best_kernel_est_list = []

# %%
from neural_network import PsfCalibration
from tqdm import tqdm
from torch.quasirandom import SobolEngine
from algorithm import LbfgsPsfCalibration
from algorithm.main import estimate_psf_zernike_mlp

psf_calib_nn_B = PsfCalibration(num_coeffs=num_coeffs, verbose=False, **kwargs)
psf_calib_lbfgs = LbfgsPsfCalibration(psf_size=psf_size, num_coeffs=num_coeffs, **kwargs)

generator = dinv.physics.generator.DiffractionBlurGenerator(
    psf_size=psf_size, zernike_index=range(2, 2+num_coeffs), device=device
)
rel_err_sigmas = []

for sigma in sigma_list:
    print(f"Optimizing for sigma={sigma.item():.2f}")

    blurs_true = y + sigma * torch.randn_like(y)

    rel_err_sig_lbfgs = []
    rel_err_sig_nn_B = []
    rel_err_sig_nn_P = []


    progress = tqdm(range(sample_per_sigma), desc=f"Progressing for sigma={sigma.item():.2f}")
    for i in range(sample_per_sigma):

        blur_true = blurs_true[i:i+1]
        true_filter = kernels[i:i+1]


        ################# LBFGS optimization with multiple restarts #################
        coeffs_est = psf_calib_lbfgs.forward(img,
                                            blur_true,
                                            initialization_method='sobol',
                                            niter=10)
        kernels_est = psf_calib_lbfgs.generate_blur(coeffs_est)['filter']

        
        relative_error_lbfgs = (true_filter-kernels_est).abs().sum()
        rel_err_sig_lbfgs.append(relative_error_lbfgs.item())

        ################# NN REPARAMETRIZATION B #################
        coeffs_est = psf_calib_nn_B._forward_one_image(img,
                                                blur_true,
                                                niter=30,
                                                initial_coeffs=None,
                                                optimizer_type='lbfgs',
                                                crop=False)
        kernels_est = psf_calib_nn_B.kernel_generator.step(coeff=coeffs_est)['filter']
        relative_error_nn = (true_filter-kernels_est).abs().sum()
        rel_err_sig_nn_B.append(relative_error_nn.item())

        ################# NN REPARAMETRIZATION P #################
        coeffs_est = estimate_psf_zernike_mlp(img,
                                            blur_true,
                                            psf_size[0],
                                            range(2, 2+num_coeffs),
                                            verbose=False,
                                            device=device)["coefficients"]
        coeffs_est = torch.from_numpy(coeffs_est).to(**kwargs)
        kernels_est = generator.step(coeff=coeffs_est.unsqueeze(0))['filter']
        
        relative_error_nn = (true_filter-kernels_est).abs().sum()
        rel_err_sig_nn_P.append(relative_error_nn.item())

    rel_err_sigmas.append({
        "lbfgs": rel_err_sig_lbfgs,
        "nn_B": rel_err_sig_nn_B,
        "nn_P": rel_err_sig_nn_P
    })


# %%
# create pandas dataframe for plotting
import pandas as pd
df = pd.DataFrame(columns=["sigma", "method", "relative_error"])

for sigma, rel_err in zip(sigma_list, rel_err_sigmas):
    for method, rel_errors in rel_err.items():
        for iter_num, rel_error in enumerate(rel_errors):
            df.loc[len(df)] = [round(sigma.item(), 2),
                               method,
                               rel_error]

# %%
import seaborn as sns

# plot boxplot of relative error: x axis is sigma, y axis is relative error, hue is method
# latex labels for x axis
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="sigma", y="relative_error", hue="method")
plt.xlabel(r"$\sigma$", fontsize=14)
plt.ylabel(r'$\frac{\|h - \hat{h}\|}{\|h\|}$', fontsize=14)
plt.savefig(os.path.join(figure_path, "adamNN_vs_LBFGS.png"),
            dpi=300,
            bbox_inches='tight')
# %%
# %%
