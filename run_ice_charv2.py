"""
Post-experiment data analysis of PAL-XFEL ice experiment
Author: Michael Hassett
Date: 2024-06-09
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import find_peaks
from copy import deepcopy

from diffraction1d import radial_integration
from cubicity_char import cubicity, normalise_peaks
from domain_size_char import domain_size_from_gaussian
from peak_fitting import NGaussianPeaks, GaussianPeak
from paltools import Experiment, Run
from watch_data import watch_folder

plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams['savefig.dpi'] = 300


def plot_2D_average(run: Run, title: str):
    avgimg = np.zeros((2880, 2880))
    for scanid in range(1, run.numscans + 1):
        print(f"scanid {scanid}")
        for pulseid in run.getPulseIds(scanid):
            try:
                # print(pulseid)
                avgimg += (run.getImage(scanid, pulseid))
            except KeyError:
                continue
        avgimg = avgimg / run.numscans  # .mean(avgimg,axis=0)
    fig, ax = plt.subplots()
    p = ax.imshow(avgimg, norm=colors.LogNorm())
    ax.invert_yaxis()
    colorbar_axes = make_axes_locatable(ax).append_axes("right", size="10%", pad=0.1)
    fig.colorbar(p, label="intensity", cax=colorbar_axes)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_1D_average(run: Run, title: str | None = None, *, show_all: bool = False, show_first: bool = False):
    scan_id = 1
    z_dum = run.getPulseIds(scan_id)
    z_len = len(z_dum)
    x_dum, y_dum = run.getRadialAverage(scan_id, z_dum[0])
    y_len = len(y_dum)
    array_shape = [z_len, y_len]
    del x_dum, y_dum, z_dum
    data_array = np.zeros(array_shape)
    for i, pulse_id in enumerate(run.getPulseIds(scan_id)):
        xvar, yvar = run.getRadialAverage(scan_id, pulse_id)
        data_array[i] = yvar
        if show_all or (show_first and i == 0):
            fig, ax = plt.subplots()
            ax.plot(xvar, yvar)
            if title is not None:
                ax.set_title(title)
            fig.tight_layout()
            plt.show()
    fig_all, ax_all = plt.subplots()
    im = ax_all.imshow(data_array, norm=colors.LogNorm())
    ax_all.invert_yaxis()
    if title is not None:
        ax_all.set_title(title)
    colorbar_axes = make_axes_locatable(ax_all).append_axes("right", size="10%", pad=0.1)
    fig_all.colorbar(mappable=im, ax=ax_all, cax=colorbar_axes)
    fig_all.tight_layout()
    return fig_all, ax_all


def model_shot(run: Run, title=None, *, show_all: bool = False):
    peak1 = GaussianPeak(name='hex_1', amplitude=16, mean=12.55, stddev=0.04)
    peak2 = GaussianPeak(name='hex_2', amplitude=2.3, mean=13.3, stddev=0.05)
    peak3 = GaussianPeak(name='bkg', amplitude=0.5, mean=13, stddev=0.06)
    model_params = [peak1, peak2, peak3]

    x_min, x_max = (12.25, 13.5)

    scan_id = 1
    z_dum = run.getPulseIds(scan_id)
    z_len = len(z_dum)
    x_dum, y_dum = run.getRadialAverage(scan_id, z_dum[0])
    y_len = len(y_dum)
    array_shape = [z_len, y_len]
    model_array = np.zeros(array_shape)
    data_array = np.zeros(array_shape)
    peaks = []
    del x_dum, y_dum, z_dum

    for i, pulse_id in enumerate(run.getPulseIds(scan_id)):
        xvar, yvar = run.getRadialAverage(scan_id, pulse_id)
        data_array[i] = yvar
        xvarsel = (x_min < xvar) & (xvar < x_max)

        if i == 0:
            gaussians = NGaussianPeaks(model_params)

        gaussians.fit(xvar[xvarsel], yvar[xvarsel], maxiter=10000, acc=0.0001)
        if show_all:
            plt.figure()
            plt.plot(xvar[xvarsel], yvar[xvarsel], color='blue')
            plt.plot(xvar[xvarsel], gaussians.model(xvar[xvarsel]),
                     label='total', linestyle='--', color='black')
            for gaussian in gaussians:
                plt.plot(xvar[xvarsel], gaussians.model[gaussian.name](xvar[xvarsel]),
                         label=gaussian.name, linestyle='--', alpha=0.5)
            plt.show()
        model_array[i] = gaussians.model(xvar)
        peaks.append(deepcopy(gaussians))
    residual = np.abs(data_array - model_array)

    x_range_plt = [225, 275]
    data_array_xrange = data_array[:, x_range_plt[0]:x_range_plt[1]]
    model_array_xrange = model_array[:, x_range_plt[0]:x_range_plt[1]]
    cmax = np.max(
        [data_array_xrange[np.nonzero(data_array_xrange)], model_array_xrange[np.nonzero(model_array_xrange)]])
    cmin = np.min(
        [data_array_xrange[np.nonzero(data_array_xrange)], model_array_xrange[np.nonzero(model_array_xrange)]])
    if cmin < 0:
        cmin = 1e-3

    # print(cmin, cmax)
    fig_width, fig_height = plt.rcParams['figure.figsize']
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(fig_width, fig_width * 1.1))
    fig.subplots_adjust(wspace=0.1)
    if title is not None:
        fig.suptitle(title)
    im1 = ax1.imshow(data_array, norm=colors.LogNorm(vmin=cmin, vmax=cmax))
    ax1.set_xlim(x_range_plt)
    ax1.set_title('Data')
    ax1.set_ylabel('Pulse')

    ax2.imshow(model_array, norm=colors.LogNorm(vmin=cmin, vmax=cmax))
    ax2.set_title('Model')
    colorbar_axes2 = make_axes_locatable(ax2).append_axes("right", size="10%", pad=0.1)
    fig.colorbar(mappable=im1, ax=ax2, cax=colorbar_axes2)

    cmin_residual = np.min(residual[:, x_range_plt[0]:x_range_plt[1]])
    cmax_residual = np.max(residual[:, x_range_plt[0]:x_range_plt[1]])
    im3 = ax3.imshow(residual, norm=colors.LogNorm(vmin=cmin_residual, vmax=cmax_residual))
    ax3.set_title('Residual')
    colorbar_axes3 = make_axes_locatable(ax3).append_axes("right", size="10%", pad=0.1)
    fig.colorbar(mappable=im3, ax=ax3, cax=colorbar_axes3)
    fig.tight_layout()
    # print(cmax_residual / cmax)
    return fig, (ax1, ax2, ax3), peaks


def plot_raw_gaussian_info(gaussians: list[NGaussianPeaks], title=None):
    fig, (ax_amps, ax_means) = plt.subplots(2, 1, sharex=True)
    y_max_amps = 0
    names = gaussians[0].names
    for name in names:
        peak_amps = [peak[name].amplitude.value for peak in gaussians]
        peak_means = [peak[name].mean.value for peak in gaussians]
        ax_amps.plot(peak_amps, label=name)
        ax_means.plot(peak_means)
        y_max_amps_new = np.max(peak_amps) * 1.1
        if y_max_amps_new > y_max_amps:
            y_max_amps = y_max_amps_new

    ax_amps.set_ylim(0, y_max_amps)
    ax_amps.set_ylabel("Intensity")

    ax_means.set_ylabel("Mean of peak")
    ax_means.set_xlabel("Pulse")
    fig.legend()
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, (ax_amps, ax_means)


def plot_cubicity(ax, gaussians: list[NGaussianPeaks]):
    cubicity_vals = [cubicity(peaks) for peaks in
                     [(gaussian['hex_1'].amplitude.value, gaussian['hex_2'].amplitude.value) for gaussian in gaussians]]
    ax.plot(cubicity_vals)
    ax.set_ylabel("Cubicity")


def plot_domain_size(ax, gaussians: list[NGaussianPeaks]):
    domain_size_vals = [domain_size_from_gaussian(gaussian['hex_1'], current_run) for gaussian in gaussians]
    ax.plot(domain_size_vals)
    ax.set_ylabel("Domain size")


def plot_cubicity_domain_size(gaussians: list[NGaussianPeaks], title=None):
    fig, (ax_cub, ax_domain) = plt.subplots(2, 1, sharex=True)
    if title is not None:
        fig.suptitle(title)
    plot_cubicity(ax_cub, gaussians)
    plot_domain_size(ax_domain, gaussians)
    ax_domain.set_xlabel("Pulse")
    fig.tight_layout()
    return fig, (ax_cub, ax_domain)


if __name__ == '__main__':
    root_path = Path('F:\gspark_snu PAL-XFEL data/ctbas/ue_240330_FXL/scan/')
    current_exp = Experiment(experiment_id="2023-2nd-XSS-040",
                             photon_energy=15e3,
                             detector_distance=0.321,
                             pixel_size=0.255 / 5760,
                             root_path=root_path)
    run_root = 'day3_rus3_shot1'
    output_root = Path(f'output/{run_root}')

    runs = root_path.glob(f'{run_root}_*_DIR')
    for run in runs:
        current_run = Run(current_exp, run.name)
        output_path = output_root / run.name
        output_path.mkdir(exist_ok=True, parents=True)
        # Plot 2D average for run
        #fig_2d, ax_2d = plot_2D_average(current_run, f"shot average {current_run.name}")

        plot_1D_average(current_run, title=run.name, show_first=True)
        plt.show()

        # Model Gaussian peaks to data
        fig_model_compare, _, gaussians = model_shot(current_run, title=run.name)  #, show_all=True)
        plt.close(fig_model_compare)
        normal_amps = np.zeros([len(gaussians), 3])
        theor_normal_amp = normalise_peaks([17.491, 9.316, 10.188])[1]
        for i, gaussian in enumerate(gaussians):
            amps = [peak.amplitude.value for peak in gaussian]
            normal_amps[i] = normalise_peaks(amps)
        plt.figure()
        plt.plot(np.full_like(normal_amps[:, 1], theor_normal_amp), 'k',
                 linestyle='dashed', label='Fully hexagonal')
        plt.plot(normal_amps[:, 1], label='Data')
        text_string = (f'Theor. [002] min: {theor_normal_amp:.4f}\n'
                       f'Data [002] average: {normal_amps[:, 1].mean():.4f}')
        plt.text(0.2, 0.8, text_string,
                 transform=plt.gca().transAxes, fontsize='large')
        plt.ylim(0, 0.6)
        plt.title(run.name)
        plt.legend()
        plt.tight_layout()
        plt.show()
        '''
        fig_model_params, _ = plot_raw_gaussian_info(gaussians, title=run.name)
        fig_cubicity_domain, _ = plot_cubicity_domain_size(gaussians, title=run.name)
        

        fig_model_compare.savefig(output_path / 'gaussian_model.png')
        fig_model_params.savefig(output_path / 'gaussian_params.png')
        fig_cubicity_domain.savefig(output_path / 'cubicity+domain.png')
        # plt.show()
        plt.close('all')
        '''
