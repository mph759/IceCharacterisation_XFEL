"""
Generic tools for reading data during PAL-XFEL beamtimes
Written for use at PAL-XFEL experiment 2023-2nd-XSS-040
Created on 2024-03-30 by Sebastian Cardoch
"""
import h5py
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt


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
    def id(self): return self.__id__

    @property
    def photon_energy(self): return self.__photon_energy__

    @property
    def wavelength(self): return self.__wavelength__

    @property
    def detector_distance(self): return self.__detector_distance__

    @property
    def path(self): return self.__root_path__


class run:
    def __init__(self, experiment: Experiment, runname: str):
        self.experiment = experiment
        self.runname = self.experiment.path / f'{runname}/'
        self.pulseinfo_filename = self.runname / "pulseInfo/"
        self.twotheta_filename = self.runname / "eh1rayMXAI_tth/"
        self.intensity_filename = self.runname / "eh1rayMXAI_int/"
        self.total_sum_filename = self.runname / "ohqbpm2_totalsum/"
        self.image_filename = self.runname / "eh1rayMX_img/"

    def getPulseIds(self, scanId):
        # if type(scanId) == list:
        #     pulseIds = np.array([self.getPulseIds(id) for id in scanId])
        #     return np.squeeze(pulseIds)

        if self.experiment.id == '2023-2nd-XSS-040':
            file_name = str("00000001_%08d.h5" % (scanId * 300))
        else:
            file_name = str("001_001_%03d.h5"%scanId)
        file = self.pulseinfo_filename / file_name
        with h5py.File(file) as f:
            pulseIds = np.array(list(f.keys()))
        return pulseIds

    def getIntensityNorm(self, scanId, pulseId):
        if np.ndim(pulseId) != 0:
            intensity_norm = np.array([self.getIntensityNorm(scanId, id) for id in pulseId])
            return np.squeeze(intensity_norm)

        if self.experiment.id == '2023-2nd-XSS-040':
            filename = self.total_sum_filename / str("00000001_%08d.h5" % (scanId * 300))
        else:
            filename = self.total_sum / str("001_001_%03d.h5" % scanId)
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

        if self.experiment.id == '2023-2nd-XSS-040':
            scan_name = "00000001_%08d.h5" % (scanId * 300)
        else:
            scan_name = "001_001_%03d.h5"%scanId
        intensity_filename = self.intensity_filename / scan_name
        yvar = run.__getRadial(intensity_filename, pulseId)
        return yvar

    def getRadialIntensityNorm(self, scanId, pulseId):
        yvar = self.getRadialIntensity(scanId, pulseId)
        norm = self.getIntensityNorm(scanId, pulseId)
        return yvar / norm

    def getRadialtwoTheta(self, scanId, pulseId):
        if np.ndim(pulseId) != 0:
            yvar = np.array([self.getRadialtwoTheta(scanId, id) for id in pulseId])
            return np.squeeze(yvar)

        if self.experiment.id == '2023-2nd-XSS-040':
            scan_name = "00000001_%08d.h5" % (scanId * 300)
        else:
            scan_name = "001_001_%03d.h5" % scanId
        twotheta_filename = self.twotheta_filename / scan_name
        xvar = run.__getRadial(twotheta_filename, pulseId)
        return xvar

    def getImage(self, scanId, pulseId):
        scan_name = "00000001_%08d.h5" % (scanId * 300)
        filename = self.image_filename / scan_name
        with h5py.File(filename) as f:
            return f[pulseId][()]

    @staticmethod
    def __getRadial(filename, pulseId):
        with h5py.File(filename) as f:
            radial = f[pulseId][()]
        return radial


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
