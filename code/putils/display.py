"""
Some useful utilities for displaying
"""

import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
import torchvision.transforms as transforms
from typing import List, Optional, Tuple, Union
import warnings

warnings.filterwarnings("ignore", module="matplotlib\..*")

_TensorArray = Union[Tensor, np.ndarray]


def update_plt_params(dict=None):
    default_params = {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": "\\usepackage{times, bm}",
        "figure.figsize": (4, 2.5),
        "figure.constrained_layout.use": True,
        "figure.autolayout": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.015,
        "font.size": 13,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.titlesize": 11,
    }
    plt.rcParams.update(default_params)
    if dict is not None:
        plt.rcParams.update(dict)


def show(imgs: List):
    """
    Show image or list of images
    """
    if not isinstance(imgs, list):
        imgs = [imgs]

    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)

    for i, img in enumerate(imgs):
        img = img.detach()
        mode = "RGB" if img.shape[0] == 3 else "L"
        img = transforms.functional.to_pil_image(img, mode)
        if mode == "RGB":
            axs[0, i].imshow(np.asarray(img))
        else:
            axs[0, i].imshow(np.asarray(img), cmap="gray")
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def show_images(
    imgs: Union[_TensorArray, List[_TensorArray], Tuple[_TensorArray]],
    title: List[str] = None,
    suptitle: str = None,
    ncols: int = None,
    colorbar: Optional[bool] = False,
    cmap: str = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    savename: Optional[str] = None,
    figsize: Optional[int] = 3,
    interpolation: Optional[str] = None,
):

    if isinstance(imgs, List) or isinstance(imgs, Tuple):
        if isinstance(imgs[0], np.ndarray):
            imgs = np.concatenate(imgs, axis=0)
        elif isinstance(imgs[0], torch.Tensor):
            imgs = to_numpy_image(torch.cat(imgs, dim=0))
    else:
        imgs = to_numpy_image(imgs)

    if imgs.ndim == 3:
        H, W, C = imgs.shape
        B = 1
    elif imgs.ndim == 4:
        B, H, W, C = imgs.shape

    if (ncols is not None) and (B % ncols == 0):
        nrows = B // ncols
    else:
        nrows = 1
        ncols = B
    offset = 0 if not colorbar else 0.75
    fig = plt.figure(figsize=(ncols * (figsize + offset), nrows * figsize))
    gs = gridspec.GridSpec(ncols=ncols, nrows=nrows)
    gs.update(wspace=0.025, hspace=0.025)

    if cmap is None:
        cmap = "gray" if C == 1 else None
    i = 0

    if title is None:
        title = [""]
    if len(title) != B:
        title += [""] * (B - len(title))
    with warnings.catch_warnings():
        if nrows > 1 and ncols > 1:
            for r in range(nrows):
                for c in range(ncols):
                    axs = plt.subplot(gs[r, c])
                    im = axs.imshow(
                        imgs[i],
                        vmin=vmin,
                        vmax=vmax,
                        cmap=cmap,
                        interpolation=interpolation,
                    )
                    axs.set_xticks([])
                    axs.set_yticks([])
                    if colorbar:
                        divider = make_axes_locatable(axs)
                        cax = divider.append_axes("right", size="3%", pad=0.1)
                        plt.colorbar(im, cax=cax)
                    if title:
                        axs.set_title(title[i], pad=3)

                    i += 1
        elif nrows > 1 or ncols > 1:
            for c in range(max(ncols, nrows)):
                axs = plt.subplot(gs[c])
                im = axs.imshow(
                    imgs[c],
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    interpolation=interpolation,
                )
                axs.set_xticks([])
                axs.set_yticks([])
                if colorbar:
                    divider = make_axes_locatable(axs)
                    cax = divider.append_axes("right", size="3%", pad=0.1)
                    plt.colorbar(im, cax=cax)
                if title:
                    axs.set_title(title[c], pad=3)

        else:
            axs = plt.subplot(gs[0])
            im = axs.imshow(
                imgs.squeeze(),
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                interpolation=interpolation,
            )
            axs.set_xticks([])
            axs.set_yticks([])
            if colorbar:
                divider = make_axes_locatable(axs)
                cax = divider.append_axes("right", size="3%", pad=0.1)
                plt.colorbar(im, cax=cax)
            if title:
                axs.set_title(title[0], pad=3)

        if suptitle is not None:
            fig.suptitle(suptitle, y=1.03)
        if savename is not None:
            fig.savefig(savename, bbox_inches="tight")  # , dpi=1200)
            plt.close()
        else:
            plt.show()


def make_grid_images(
    imgs: Union[_TensorArray, List[_TensorArray], Tuple[_TensorArray]],
    title: List[str] = None,
    suptitle: str = None,
    ncols: int = None,
    colorbar: Optional[bool] = False,
    cmap: str = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Optional[int] = 3,
):

    if isinstance(imgs, List) or isinstance(imgs, Tuple):
        if isinstance(imgs[0], np.ndarray):
            imgs = np.concatenate(imgs, axis=0)
        elif isinstance(imgs[0], torch.Tensor):
            imgs = to_numpy_image(torch.cat(imgs, dim=0))
    else:
        imgs = to_numpy_image(imgs)

    if imgs.ndim == 3:
        H, W, C = imgs.shape
        B = 1
    elif imgs.ndim == 4:
        B, H, W, C = imgs.shape

    if (ncols is not None) and (B % ncols == 0):
        nrows = B // ncols
    else:
        nrows = 1
        ncols = B
    fig = plt.figure(figsize=(ncols * figsize, nrows * figsize))
    gs = gridspec.GridSpec(ncols=ncols, nrows=nrows)
    gs.update(wspace=0.025, hspace=0.025)

    if cmap is None:
        cmap = "gray" if C == 1 else None
    i = 0

    if title is None:
        title = [""]
    if len(title) != B:
        title += [""] * (B - len(title))
    if nrows > 1 and ncols > 1:
        for r in range(nrows):
            for c in range(ncols):
                axs = plt.subplot(gs[r, c])
                im = axs.imshow(
                    imgs[i], vmin=vmin, vmax=vmax, cmap=cmap, interpolation="none"
                )
                axs.set_xticks([])
                axs.set_yticks([])
                if colorbar:
                    divider = make_axes_locatable(axs)
                    cax = divider.append_axes("right", size="3%", pad=0.1)
                    plt.colorbar(im, cax=cax)
                if title:
                    axs.set_title(title[i])
                i += 1
    elif nrows > 1 or ncols > 1:
        for c in range(max(ncols, nrows)):
            axs = plt.subplot(gs[c])
            im = axs.imshow(
                imgs[c], vmin=vmin, vmax=vmax, cmap=cmap, interpolation="none"
            )
            axs.set_xticks([])
            axs.set_yticks([])
            if colorbar:
                divider = make_axes_locatable(axs)
                cax = divider.append_axes("right", size="3%", pad=0.1)
                plt.colorbar(im, cax=cax)
            if title:
                axs.set_title(title[c], pad=30)

    else:
        axs = plt.subplot(gs[0])
        im = axs.imshow(
            imgs.squeeze(), vmin=vmin, vmax=vmax, cmap=cmap, interpolation="none"
        )
        axs.set_xticks([])
        axs.set_yticks([])
        if colorbar:
            divider = make_axes_locatable(axs)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            plt.colorbar(im, cax=cax)
        if title:
            axs.set_title(title[0])

    if suptitle is not None:
        fig.suptitle(suptitle)

    return fig


def to_numpy_image(input):
    if isinstance(input, Tensor):
        if input.dim() == 3:
            return input.detach().cpu().permute(1, 2, 0).numpy()
        elif input.dim() == 4:
            return input.detach().cpu().permute(0, 2, 3, 1).numpy()

    elif isinstance(input, np.ndarray):
        print("Warning: input is already a numpy array")
        return input
    else:
        raise ValueError(f"Cannot convert {type(input)} to numpy image")