"""
Domain size characterisation of ice diffraction peaks from XFEL experiments
Written for use at PAL-XFEL experiment 2023-2nd-XSS-040
Created on 2024-03-27 by Michael Hassett
"""
import numpy as np
from peak_fitting import GaussianPeak
from paltools import Run, Experiment

def domain_size(twotheta: float, fwhm: float, wavelength: float, K: float=0.94) -> float:
    """
    Calculate the domain size of an ice crystal from twotheta and fwhm of diffraction peak
    :param twotheta: 2theta angle in degrees of diffraction peak
    :param fwhm: Full-width half maximum of diffraction peak
    :param wavelength: Wavelength of experimental set up (in metres preferred)
    :param K: Characteristic parameter of phase (default 0.94 for spherical cubic)
    :return:
    """
    return (K * wavelength) / (fwhm * np.cos(np.deg2rad(twotheta)/2))

def domain_size_from_gaussian(gaussian: GaussianPeak, run: Run):
    return domain_size(twotheta=gaussian.mean.value, fwhm=gaussian.fwhm, wavelength=run.experiment.wavelength.value)


def domain_size_testing():
    K = 0.94                # Characteristic parameter of spherical cubic crystals
    wavelength = 0.082E-9   # Experimental wavelength in metres
    print(domain_size(twotheta=26.02, fwhm=0.00244, wavelength=wavelength, K=K))


if __name__ == '__main__':
    domain_size_testing()
