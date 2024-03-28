"""
Peak fitting for hexagonal and cubic ice diffraction peaks from XFEL experiments
Written for use at PAL-XFEL experiment 2023-2nd-XSS-040
Created on 2024-03-26 by Michael Hassett
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Gaussian1D


class CubicIceFitting:
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


class HexIceFitting(CubicIceFitting):
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
    # Generate simulated diffraction data (not realistic) from Gaussian curves,
    # adding random steps away to simulation diffusion
    m = Gaussian1D(amplitude=10, mean=20, stddev=5) + Gaussian1D(amplitude=10, mean=50, stddev=5) + Gaussian1D(
        amplitude=10, mean=80, stddev=5)
    x = np.linspace(0, 100, 2000)
    data = m(x)
    data = data + np.sqrt(data) * (np.random.random(x.size) - 0.5)
    # data -= data.min()
    plt.plot(x, data)

    # Instantiate the HexIceFitting model, with approximate
    # amplitude, mean, and standard deviations for the modelled peaks
    hex_ice = HexIceFitting(amplitude=[15, 8, 12], mean=[15, 52, 79], stddev=[4, 5, 6])
    hex_ice.fit(x, data)

    print(f'Mean: {hex_ice.peak1.mean.value}, Amplitude: {hex_ice.peak1.amplitude.value}, FWHM: {hex_ice.peak1.fwhm}')
    print(f'Mean: {hex_ice.peak2.mean.value}, Amplitude: {hex_ice.peak2.amplitude.value}, FWHM: {hex_ice.peak2.fwhm}')
    print(f'Mean: {hex_ice.peak3.mean.value}, Amplitude: {hex_ice.peak3.amplitude.value}, FWHM: {hex_ice.peak3.fwhm}')

    plt.plot(x, hex_ice.model(x))

    cubic_ice = CubicIceFitting(amplitude=15, mean=52, stddev=5)
    cubic_ice.fit(x, data)

    print(f'Mean: {cubic_ice.peak1.mean.value},'
          f'Amplitude: {cubic_ice.peak1.amplitude.value},'
          f'FWHM: {cubic_ice.peak1.fwhm}')

    plt.plot(x, cubic_ice.model(x))

    plt.show()


if __name__ == '__main__':
    gaussian_fitting_testing()
