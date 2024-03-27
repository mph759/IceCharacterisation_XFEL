"""
Peak fitting for hexagonal and cubic ice diffraction peaks from XFEL experiments
Written for use at PAL-XFEL experiment 2023-2nd-XSS-040
Created on 2024-03-26 by Michael Hassett
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Gaussian1D


class CubicIceModel:
    def __init__(self, amplitude: int, mean: int, stddev: float):
        self.fitter = LevMarLSQFitter()
        self.__model__ = self.__init_model__(amplitude, mean, stddev)

    def __init_model__(self, amplitude: int, mean: int, stddev: float):
        return Gaussian1D(amplitude=amplitude, mean=mean, stddev=stddev, name='cubic')

    def fit(self, x_data: list[int], y_data: list[float]):
        self.__model__ = self.fitter(self.model, x_data, y_data)

    @property
    def model(self):
        return self.__model__

    @property
    def peak1(self):
        return self.__model__


class HexIceModel(CubicIceModel):
    def __init__(self,
                 amplitude: tuple[int, int, int],
                 mean: tuple[int, int, int],
                 stddev: tuple[int, int, int]):
        super().__init__(amplitude, mean, stddev)

    def __init_model__(self, amplitude: tuple[int, int, int],
                       mean: tuple[int, int, int],
                       stddev: tuple[float, float, float]):
        self.__hex_1__ = Gaussian1D(amplitude=amplitude[0], mean=mean[0], stddev=stddev[0],
                                name='hex_1')
        self.__hex_2__ = Gaussian1D(amplitude=amplitude[1], mean=mean[1], stddev=stddev[1],
                                name='hex_2')
        self.__hex_3__ = Gaussian1D(amplitude=amplitude[2], mean=mean[2], stddev=stddev[2],
                                name='hex_3')
        return self.__hex_1__ + self.__hex_2__ + self.__hex_3__

    @property
    def peak1(self):
        return self.__model__['hex_1']

    @property
    def peak2(self):
        return self.__model__['hex_2']

    @property
    def peak3(self):
        return self.__model__['hex_3']


def gaussian_fitting_testing():
    m = Gaussian1D(amplitude=10, mean=20, stddev=5) + Gaussian1D(amplitude=10, mean=50, stddev=5) + Gaussian1D(
        amplitude=10, mean=80, stddev=5)
    x = np.linspace(0, 100, 2000)
    data = m(x)
    data = data + np.sqrt(data) * (np.random.random(x.size) - 0.5)
    # data -= data.min()
    plt.plot(x, data)

    HexIce = HexIceModel(amplitude=[15, 8, 12], mean=[15, 52, 79], stddev=[4, 5, 6])
    HexIce.fit(x, data)

    print(f'Mean: {HexIce.peak1.mean.value}, Amplitude: {HexIce.peak1.amplitude.value}, FWHM: {HexIce.peak1.fwhm}')
    print(f'Mean: {HexIce.peak2.mean.value}, Amplitude: {HexIce.peak2.amplitude.value}, FWHM: {HexIce.peak2.fwhm}')
    print(f'Mean: {HexIce.peak3.mean.value}, Amplitude: {HexIce.peak3.amplitude.value}, FWHM: {HexIce.peak3.fwhm}')

    plt.plot(x, HexIce.model(x))

    CubicIce = CubicIceModel(amplitude=15, mean=52, stddev=5)
    CubicIce.fit(x, data)

    print(f'Mean: {CubicIce.peak1.mean.value}, Amplitude: {CubicIce.peak1.amplitude.value}, FWHM: {CubicIce.peak1.fwhm}')

    plt.plot(x, CubicIce.model(x))

    plt.show()


if __name__ == '__main__':
    gaussian_fitting_testing()
