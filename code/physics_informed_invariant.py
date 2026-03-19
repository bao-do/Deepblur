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



# %%

objective_fn = LossFidelity(reduction="sum",
                            norm="l2",
                            physics=conv2d_fft,
                            **kwargs)


kernel_est_generator = DiffractionBlurGenerator(psf_size=psf_size,
                                            max_zernike_amplitude=0.3,
                                            zernike_index=range(2,2+num_coeffs),
                                            num_channels=1,
                                            **kwargs)
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
from evotorch import Problem
from evotorch.algorithms import CMAES

psfcalib = PsfCalibration(num_coeffs=num_coeffs, verbose=False, **kwargs)

rel_err_sigmas = []

for sigma in sigma_list:
    print(f"Optimizing for sigma={sigma.item():.2f}")

    blurs_true = y + sigma * torch.randn_like(y)

    rel_err_sig_lbfgs = []
    rel_err_sig_nn = []
    rel_err_sig_cmaes = []

    progress = tqdm(range(sample_per_sigma), desc=f"Progressing for sigma={sigma.item():.2f}")
    for i in range(sample_per_sigma):

        blur_true = blurs_true[i:i+1]
        true_filter = kernels[i:i+1]


        ################# LBFGS optimization with multiple restarts #################
        best_loss = float('inf')
        best_loss_iter = None
        best_kernel_est = None
        
        # initialize inital points covering the search space, using sobol sequence
        coeffs_restarts = SobolEngine(dimension=num_coeffs,
                                      scramble=True,
                                      seed=random_seed()).draw(n_restarts).to(**kwargs)
        coeffs_restarts = max_zernike_amplitude * (coeffs_restarts - 0.5)

        for restart in range(n_restarts):
            # coeffs = kernel_est_generator.step(coeff=coeffs_restarts[restart].unsqueeze(0)
            #                                    )['coeff'].requires_grad_(True)
            coeffs = coeffs_restarts[restart].unsqueeze(0).requires_grad_(True)
            optimizer = LBFGS([coeffs],
                            lr=1.0,
                            history_size=10,
                            max_iter=20,
                            line_search_fn="strong_wolfe")
            loss_iter = []
            for i in range(niter):
                
                def closure():
                    filters = kernel_est_generator.step(batch_size=1,
                                                    coeff=coeffs)['filter']
                    optimizer.zero_grad()
                    loss = objective_fn(img, blur_true, filter=filters, crop=False)
                    loss.backward()
                    return loss
                optimizer.step(closure)
                loss = closure()
                loss_iter.append(loss.item())

            

            if loss_iter[-1] < best_loss:
                best_loss = loss_iter[-1]
                best_loss_iter = loss_iter
                best_kernel_est = kernel_est_generator.step(batch_size=1,
                                                        coeff=coeffs)['filter'].detach()
        
        relative_error_lbfgs = torch.norm(true_filter-best_kernel_est)/torch.norm(true_filter)
        rel_err_sig_lbfgs.append(relative_error_lbfgs.item())

        ################# NN REPARAMETRIZATION #################
        blur_est, _ = psfcalib._forward_one_image(img,
                                                blur_true,
                                                niter=10,
                                                initial_coeffs=None,
                                                optimizer_type='lbfgs',
                                                crop=False)
        relative_error_nn = torch.norm(true_filter-blur_est['filter'])/torch.norm(true_filter)
        rel_err_sig_nn.append(relative_error_nn.item())
        
        ############### CAMA-ES optimization ###############
        def objective_func_wrapper(coeffs):
            filters = kernel_est_generator.step(coeff=coeffs.unsqueeze(0))['filter']
            loss = objective_fn(x=img,
                                y=blur_true,
                                filter=filters,
                                crop=False).item() 
            return loss

        problem = Problem(
            "min", 
            objective_func=objective_func_wrapper, 
            solution_length=num_coeffs, 
            initial_bounds=(-0.15, 0.15),
            **kwargs
        )
        searcher = CMAES(problem, stdev_init=0.001)
        searcher.run(num_generations=200)
        best_weights = searcher.status["pop_best"].values.clone()
        kernel_est = kernel_est_generator.step(coeff=best_weights.unsqueeze(0))['filter']

        relative_error_cmaes = torch.norm(kernel_est-true_filter)/torch.norm(true_filter)
        rel_err_sig_cmaes.append(relative_error_cmaes.item())

    rel_err_sigmas.append({
        "lbfgs": rel_err_sig_lbfgs,
        "nn": rel_err_sig_nn,
        "cmaes": rel_err_sig_cmaes
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
