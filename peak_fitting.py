import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Gaussian1D


class CubicIceModel:
    def __init__(self, amplitude, mean, stddev):
        self.init_amplitude = amplitude
        self.init_mean = mean
        self.init_stddev = stddev
        self.init_model = Gaussian1D(amplitude=10, mean=12, stddev=5, name='cubic')
        self.fitter = LevMarLSQFitter()


class HexIceModel:
    def __init__(self,
                 amplitudes: tuple[int, int, int],
                 means: tuple[int, int, int],
                 stddevs: tuple[int, int, int]):
        self.init_amplitude = amplitudes
        self.init_mean = means
        self.init_stddev = stddevs

        self.amplitudes = [None, None, None]
        self.means = [None, None, None]
        self.stddevs = [None, None, None]
        self.fwhm = [None, None, None]

        hex_1 = Gaussian1D(amplitude=self.init_amplitude[0], mean=self.init_mean[0], stddev=self.init_stddev[0],
                           name='hex_1')
        hex_2 = Gaussian1D(amplitude=self.init_amplitude[1], mean=self.init_mean[1], stddev=self.init_stddev[1],
                           name='hex_2')
        hex_3 = Gaussian1D(amplitude=self.init_amplitude[2], mean=self.init_mean[2], stddev=self.init_stddev[2],
                           name='hex_3')
        self.init_model = hex_1 + hex_2 + hex_3
        self.fitter = LevMarLSQFitter()
        self.fitted_model = None

    def fit(self, x_data, y_data):
        self.fitted_model = self.fitter(self.init_model, x_data, y_data)
        self.__get_amplitudes__()

    def __get_amplitudes__(self):
        hex_1_model = self.fitted_model['hex_1']
        hex_2_model = self.fitted_model['hex_2']
        hex_3_model = self.fitted_model['hex_3']
        for i, model in enumerate([hex_1_model, hex_2_model, hex_3_model]):
            self.amplitudes[i] = model.amplitude
            self.means[i] = model.mean
            self.stddevs[i] = model.stddev
            self.fwhm[i] = model.fwhm


def gaussian_fitting():
    m = Gaussian1D(amplitude=10, mean=10, stddev=5) + Gaussian1D(amplitude=10, mean=50, stddev=5) + Gaussian1D(
        amplitude=10, mean=80, stddev=5)
    x = np.linspace(0, 200, 2000)
    data = m(x)
    data = data + np.sqrt(data) * (np.random.random(x.size) - 0.5)
    # data -= data.min()
    plt.plot(x, data)

    HexIce = HexIceModel(amplitudes=[10, 10, 10], means=[10, 50, 80], stddevs=[5, 5, 5])
    HexIce.fit(x, data)

    print('Amplitudes')
    for amplitude in HexIce.amplitudes:
        print(amplitude)

    print('FWHM')
    for fwhm in HexIce.fwhm:
        print(fwhm)

    plt.plot(x, data)
    plt.plot(x, HexIce.fitted_model(x))
    plt.show()


if __name__ == '__main__':
    gaussian_fitting()
