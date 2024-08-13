"""
Peak fitting for hexagonal and cubic ice diffraction peaks from XFEL experiments
Written for use at PAL-XFEL experiment 2023-2nd-XSS-040
Created on 2024-03-26 by Michael Hassett
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Gaussian1D, Voigt1D
from astropy import units as u
from typing_extensions import deprecated

from diffract_io import diffraction_h5
from dataclasses import dataclass


class VoigtPeak(Voigt1D):
    def bind_mean(self, min_value=None, max_value=None):
        if min_value and max_value is None:
            self.bounds['x_0'] = [self.x_0 - self.fwhm_L, self.x_0 + self.fwhm_L]
        elif min_value is not None and max_value is None:
            self.bounds['x_0'] = [self.x_0 - min_value, self.x_0 + min_value]
        else:
            self.bounds['x_0'] = [min_value, max_value]

    @property
    def mean(self):
        return self.x_0

    @property
    def amplitude(self):
        return self.amplitude_L

    @property
    def fwhm(self):
        return self.fwhm_G / np.sqrt(1 + np.pi ** 2 * np.log(2) * (self.fwhm_L / self.fwhm_G)**2)


class GaussianPeak(Gaussian1D):
    def bind_mean(self, min_value=None, max_value=None):
        if min_value and max_value is None:
            self.bounds['mean'] = [self.mean - self.fwhm, self.mean + self.fwhm]
        elif min_value is not None and max_value is None:
            self.bounds['mean'] = [self.mean - min_value, self.mean + min_value]
        else:
            self.bounds['mean'] = [min_value, max_value]


class NPeaks:
    def __init__(self, models: list[GaussianPeak] | list[VoigtPeak]):
        self.num_models = len(models)
        self.fitter = LevMarLSQFitter()
        self.__model__ = np.sum(models)

    def fit(self, x_data: list, y_data: list, *args, **kwargs):
        self.__model__ = self.fitter(self.model, x_data, y_data, *args, **kwargs)

    @property
    def model(self):
        return self.__model__

    @property
    def names(self):
        return list(self.__model__.submodel_names)

    def __getitem__(self, name):
        return self.model[name]


class CubicIceFitting:
    def __init__(self, amplitude: int,
                 mean: int,
                 stddev: float,
                 name: str = None,
                 *args, **kwargs):
        self.fitter = LevMarLSQFitter(*args, **kwargs)
        if name is None:
            self.__model__ = self.__init_model__(amplitude, mean, stddev)
        else:
            self.__model__ = self.__init_model__(amplitude, mean, stddev, name=name)
        __peaks__ = {0: name}

    def __init_model__(self, amplitude: int, mean: int, stddev: float, name: str = 'cubic'):
        return Gaussian1D(amplitude=amplitude, mean=mean, stddev=stddev, name=name)

    def fit(self, x_data: list, y_data: list):
        self.__model__ = self.fitter(self.model, x_data, y_data)

    @property
    def model(self):
        return self.__model__

    def model_data(self, index):
        if index == 0:
            name = 'cubic'
        else:
            raise ValueError('Peak index out of range')
        return self.__parse_fit__(name)

    def __parse_fit__(self, name):
        try:
            peak = self.model[name]
        except TypeError:
            peak = self.model
        return {"name": name,
                "amplitude": peak.amplitude.value,
                "mean": peak.mean.value,
                "stddev": peak.stddev.value,
                "FWHM": peak.fwhm}

    @property
    def peak1(self):
        return self.__model__


class HexIceFitting(CubicIceFitting):
    def __init__(self,
                 amplitude: tuple,
                 mean: tuple,
                 stddev: tuple,
                 name: str = None,
                 *args, **kwargs):
        super().__init__(amplitude, mean, stddev, name, *args, **kwargs)

    def __init_model__(self, amplitude: tuple,
                       mean: tuple,
                       stddev: tuple,
                       name: list = ['hex_1', 'hex_2', 'hex_3']):
        self.peak_names = name
        self.__peak1__ = Gaussian1D(amplitude=amplitude[0], mean=mean[0], stddev=stddev[0],
                                    name=name[0])
        self.__peak2__ = Gaussian1D(amplitude=amplitude[1], mean=mean[1], stddev=stddev[1],
                                    name=name[1])
        self.__peak3__ = Gaussian1D(amplitude=amplitude[2], mean=mean[2], stddev=stddev[2],
                                    name=name[2])
        return self.__peak1__ + self.__peak2__ + self.__peak3__

    @property
    def peak1(self):
        return self.__model__[self.peak_names[0]]

    @property
    def peak2(self):
        return self.__model__[self.peak_names[1]]

    @property
    def peak3(self):
        return self.__model__[self.peak_names[2]]

    def model_data(self, index):
        name = self.peak_names[index]
        return self.__parse_fit__(name)


def gaussian_fitting_testing():
    # Generate simulated diffraction data (not realistic) from Gaussian curves,
    # adding random steps away to simulation diffusion
    m = (Gaussian1D(amplitude=10, mean=20, stddev=5) +
         Gaussian1D(amplitude=10, mean=50, stddev=5) +
         Gaussian1D(amplitude=10, mean=80, stddev=5))
    x = np.linspace(0, 100, 2000)
    data = m(x)
    data = data + np.sqrt(data) * (np.random.random(x.size) - 0.5)
    # data -= data.min()
    plt.plot(x, data)

    # Instantiate the HexIceFitting model, with approximate
    # amplitude, mean, and standard deviations for the modelled peaks
    hex_ice = HexIceFitting(amplitude=[15, 8, 12], mean=[15, 52, 79], stddev=[4, 5, 6])
    hex_ice.fit(x, data)

    for i in range(0, 3):
        peak = hex_ice.model_data(i)
        for key, item in peak.items():
            print(f'{key}: {item}')
        print('\n')
    plt.plot(x, hex_ice.model(x))

    cubic_ice = CubicIceFitting(amplitude=15, mean=52, stddev=5)
    cubic_ice.fit(x, data)

    peak = cubic_ice.model_data(0)
    for key, item in peak.items():
        print(f'{key}: {item}')
    print('\n')

    plt.plot(x, cubic_ice.model(x))

    plt.show()


def sim_data_fitting_testing(data_path, data_name, amplitude, mean, stddev):
    q_select = [1.5, 2]
    diff_pattern = diffraction_h5(data_path, q_select)
    fig, ax = diff_pattern.plot_rad_intensity()

    hex_ice = HexIceFitting(amplitude=amplitude, mean=mean, stddev=stddev)
    hex_ice.fit(
        diff_pattern.q[(diff_pattern.q_selection[0] < diff_pattern.q) & (diff_pattern.q < diff_pattern.q_selection[1])],
        diff_pattern.rad_intensity[
            (diff_pattern.q_selection[0] < diff_pattern.q) & (diff_pattern.q < diff_pattern.q_selection[1])])

    for i in range(0, 3):
        peak = hex_ice.model_data(i)
        for key, item in peak.items():
            print(f'{key}: {item}')
        print('\n')
    ax.plot(diff_pattern.q[(diff_pattern.q_selection[0] < diff_pattern.q)
                           & (diff_pattern.q < diff_pattern.q_selection[1])],
            hex_ice.model(diff_pattern.q)[(diff_pattern.q_selection[0] < diff_pattern.q)
                                          & (diff_pattern.q < diff_pattern.q_selection[1])])
    ax.set_title(data_name)
    return hex_ice


def n_gaussian_test():
    # Generate simulated diffraction data (not realistic) from Gaussian curves,
    # adding random steps away to simulation diffusion
    m = (Gaussian1D(amplitude=10, mean=20, stddev=5) +
         Gaussian1D(amplitude=50, mean=50, stddev=5) +
         Gaussian1D(amplitude=10, mean=80, stddev=5))
    x = np.linspace(0, 100, 2000)
    data = m(x)
    data = data + np.sqrt(data) * (np.random.random(x.size) - 0.5)
    # data -= data.min()
    plt.plot(x, data, color='k', label='data')

    peak1 = GaussianPeak(name='hex_1', amplitude=10, mean=19, stddev=6)
    peak2 = GaussianPeak(name='hex_2', amplitude=10, mean=51, stddev=4)
    peak3 = GaussianPeak(name='hex_3', amplitude=10, mean=82, stddev=5)
    model_params = [peak1, peak2, peak3]

    guassians = NPeaks(model_params)
    guassians.fit(x, data, maxiter=1000, acc=0.001)
    plt.plot(x, guassians.model(x), label='full model', color='r', alpha=0.8, linestyle='dashed')
    for gaussian in model_params:
        plt.plot(x, guassians.model[gaussian.name](x), label=gaussian.name, alpha=0.5, linestyle='dashed')
        print(gaussian)
        print(gaussian.amplitude.value)
    plt.legend()


if __name__ == '__main__':
    '''
    gaussian_fitting_testing()

    
    test_path = Path().cwd() / 'test' / '1h.h5'
    test_name = 'hexagonal ice'
    sim_data_fitting_testing(test_path, test_name,
                             amplitude=[6.5e-5, 1.3e-4, 2.9e-5],
                             mean=[1.6, 1.7, 1.8],
                             stddev=[0.01, 0.01, 0.01])
    '''
    n_gaussian_test()
    plt.show()
