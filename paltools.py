"""
Generic tools for reading data during PAL-XFEL beamtimes
Written for use at PAL-XFEL experiment 2023-2nd-XSS-040
Created on 2024-03-30 by Sebastian Cardoch
"""
import h5py
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import glob
from scipy import sparse


class Experiment:
    def __init__(self, experiment_id: str,
                 photon_energy: float,
                 detector_distance: float,
                 root_path: str):
        self.__id__ = experiment_id
        self.__photon_energy__ = photon_energy
        self.__wavelength__ = energy2wavelength(self.photon_energy)
        self.__detector_distance__ = detector_distance
        if Path(root_path).exists():
            self.__root_path__ = Path(root_path)
        else:
            raise FileNotFoundError('Root path does not exist')

    @property
    def id(self):
        return self.__id__

    @property
    def photon_energy(self):
        return self.__photon_energy__

    @property
    def wavelength(self):
        return self.__wavelength__

    @property
    def detector_distance(self):
        return self.__detector_distance__

    @property
    def path(self):
        return self.__root_path__


class run:
    def __init__(self, experiment: Experiment, runname: str):
        self.experiment = experiment
        self.runname = self.experiment.path / f'{runname}/'
        self.pulseinfo_filename = self.runname / "pulseInfo/"
        self.twotheta_filename = self.runname / "eh1rayMXAI_tth/"
        self.intensity_filename = self.runname / "eh1rayMXAI_int/"
        self.total_sum_filename = self.runname / "ohqbpm2_totalsum/"
        self.image_filename = self.runname / "eh1rayMX_img/"
        self.numscans = len((self.pulseinfo_filename.glob("*.h5")))
        message = f"Found {self.numscans} scan files in run {runname}"
        print(message, end="\r")

    @staticmethod
    def getFileName(filepath, scanId):
        return sorted(Path(filepath).glob("*.h5"))[scanId - 1]

    def getPulseIds(self, scanId):
        filename = self.getFileName(self.pulseinfo_filename, scanId)
        with h5py.File(filename) as f:
            pulseIds = np.array(list(f.keys()))
        return pulseIds

    def getIntensityNorm(self, scanId, pulseId):
        if np.ndim(pulseId) != 0:
            intensity_norm = np.array([self.getIntensityNorm(scanId, id) for id in pulseId])
            return np.squeeze(intensity_norm)

        filename = self.getFileName(self.total_sum_filename, scanId)
        with h5py.File(filename) as f:
            return f[pulseId][()]

    def getRadialAverage(self, scanId, pulseIds):
        yvar = self.getRadialIntensity(scanId, pulseIds)
        xvar = self.getRadialtwoTheta(scanId, pulseIds)
        return xvar, yvar

    def getRadialAverageNorm(self, scanId, pulseIds):
        yvar = self.getRadialIntensityNorm(scanId, pulseIds)
        xvar = self.getRadialtwoTheta(scanId, pulseIds)
        return xvar, yvar

    def getRadialIntensity(self, scanId, pulseId):
        if np.ndim(pulseId) != 0:
            yvar = np.array([self.getRadialIntensity(scanId, id) for id in pulseId])
            return np.squeeze(yvar)

        filename = self.getFileName(self.intensity_filename, scanId)
        yvar = run.__getRadial(filename, pulseId)
        return yvar

    def getRadialIntensityNorm(self, scanId, pulseId):
        yvar = self.getRadialIntensity(scanId, pulseId)
        norm = self.getIntensityNorm(scanId, pulseId)
        return yvar / norm

    def getRadialtwoTheta(self, scanId, pulseId):
        if np.ndim(pulseId) != 0:
            yvar = np.array([self.getRadialtwoTheta(scanId, id) for id in pulseId])
            return np.squeeze(yvar)

        filename = self.getFileName(self.twotheta_filename, scanId)
        xvar = run.__getRadial(filename, pulseId)
        return xvar

    def getImage(self, scanId, pulseId):
        if np.ndim(pulseId) != 0:
            yvar = np.array([self.getImage(scanId, id) for id in pulseId])
            return np.squeeze(yvar)
        filename = self.getFileName(self.image_filename, scanId)
        with h5py.File(filename) as f:
            return f[pulseId][()]

    @staticmethod
    def __getRadial(filename, pulseId):
        with h5py.File(filename) as f:
            radial = f[pulseId][()]
        return radial


def remove_baseline(y, ratio=1e-6, lam=2000, niter=10, full_output=False):
    """removes baseline of 1d signal y using Asymmetric Least Squares Smoothing method"""
    # remove nans and adjust r vector length
    y = y[~np.isnan(y)][0:]
    # x = np.linspace(x[0],x[1],len(y))

    L = len(y)
    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2 * diag, diag], [0, -1, -2], L, L - 2)
    H = lam * D.dot(D.T)  # The transposes are flipped w.r.t the Algorithm on pg. 252
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    crit = 1
    count = 0
    while crit > ratio:
        z = sparse.linalg.spsolve(W + H, W * y)
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        w_new = 1 / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        crit = np.linalg.norm(w_new - w) / np.linalg.norm(w)
        w = w_new
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values
        count += 1
        if count > niter:
            # print('Maximum number of iterations exceeded')
            break
    if full_output:
        info = {'num_iter': count, 'stop_criterion': crit}
        return z, d, info
    else:
        return y - z


def crop_signal(y, low_threshold, high_threshold):
    """Remove low_threshold*100% from the lower end of x,y and high_threshold*100% of the higher end
    for example low_threshol=0.1, high_threshold=0.1 removes 10% of signal at both ends"""
    y = y[~np.isnan(y)][0:]
    L = len(y)
    low = int(L * (low_threshold))
    high = int(L * (1 - high_threshold))
    return y[low:high]


def partition_signal(y, noise_start, noise_end, signal_start, signal_end):
    """reads x,y and return the signal,noise regions based on input fractions (noise_start=0.1,noise_end=0.2 means we expect noise
    in interval from 10-20% of max radius)"""
    y = y[~np.isnan(y)][0:]
    L = len(y)
    noise = y[int(L * noise_start):int(L * noise_end)]
    signal = y[int(L * signal_start):int(L * signal_end)]
    return signal, noise


def snr_3(y):
    snr_signal = np.var(y) / np.mean(y)
    return snr_signal


def radial_average(imarray, step_size, skip=10):
    """radial average from the TIFF files. assumes the beam is centered in the middle of the array."""

    def nanmean(x):
        return np.NaN if np.all(x != x) else np.nanmean(x)

    size_x, size_y = imarray.shape
    x0, y0 = (size_x // 2, size_y // 2)
    x, y = np.meshgrid(np.arange(size_y), np.arange(size_x))
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    radimarray = np.array(
        [nanmean(imarray[(R <= r) & (r < R + step_size)]) for R in range(skip, imarray.shape[0], step_size)])
    r = np.linspace(skip, size_x, len(radimarray))
    return r, radimarray


def remove_baseline(y, ratio=1e-6, lam=2000, niter=10, full_output=False):
    """removes baseline of 1d signal y using Asymmetric Least Squares Smoothing method"""
    # remove nans and adjust r vector length
    y = y[~np.isnan(y)][0:]
    # x = np.linspace(x[0],x[1],len(y))

    L = len(y)
    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2 * diag, diag], [0, -1, -2], L, L - 2)
    H = lam * D.dot(D.T)  # The transposes are flipped w.r.t the Algorithm on pg. 252
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    crit = 1
    count = 0
    while crit > ratio:
        z = sparse.linalg.spsolve(W + H, W * y)
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        w_new = 1 / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        crit = np.linalg.norm(w_new - w) / np.linalg.norm(w)
        w = w_new
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values
        count += 1
        if count > niter:
            # print('Maximum number of iterations exceeded')
            break
    if full_output:
        info = {'num_iter': count, 'stop_criterion': crit}
        return z, d, info
    else:
        return y - z


def crop_signal(y, low_threshold, high_threshold):
    """Remove low_threshold*100% from the lower end of x,y and high_threshold*100% of the higher end
    for example low_threshol=0.1, high_threshold=0.1 removes 10% of signal at both ends"""
    y = y[~np.isnan(y)][0:]
    L = len(y)
    low = int(L * (low_threshold))
    high = int(L * (1 - high_threshold))
    return y[low:high]


def partition_signal(y, noise_start, noise_end, signal_start, signal_end):
    """reads x,y and return the signal,noise regions based on input fractions (noise_start=0.1,noise_end=0.2 means we expect noise
    in interval from 10-20% of max radius)"""
    y = y[~np.isnan(y)][0:]
    L = len(y)
    noise = y[int(L * noise_start):int(L * noise_end)]
    signal = y[int(L * signal_start):int(L * signal_end)]
    return signal, noise

def snr_3(y):
    snr_signal = np.var(y) / np.mean(y)
    return snr_signal


def radial_average(imarray, step_size, skip=10):
    """radial average from the TIFF files. assumes the beam is centered in the middle of the array."""

    def nanmean(x):
        return np.NaN if np.all(x != x) else np.nanmean(x)

    size_x, size_y = imarray.shape
    x0, y0 = (size_x // 2, size_y // 2)
    x, y = np.meshgrid(np.arange(size_y), np.arange(size_x))
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    radimarray = np.array(
        [nanmean(imarray[(R <= r) & (r < R + step_size)]) for R in range(skip, imarray.shape[0], step_size)])
    r = np.linspace(skip, size_x, len(radimarray))
    return r, radimarray


def loadmask(filename):
    with h5py.File("/pal/home/gspark_snu/ctbas/ue_240330_FXL/scratch/%s" % filename) as f:
        mask = f["mask"][()]
    return mask


def twothetaq(twoTheta, wavelength):
    return (1 / wavelength) * np.sin(np.deg2rad(twoTheta / 2))


def qtwotheta(q, wavelength):
    return 2 * np.arcsin(q * wavelength)


def energy2wavelength(photon_energy):
    c = 299792458  # [m/s]
    h = 6.582119569e-16  # [eV*s]
    return c * h / photon_energy


def bins2twotheta(radial_bins, detector_distance):
    return 4 * np.arctan(radial_bins / 2 * detector_distance)
