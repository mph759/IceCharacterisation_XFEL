import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Gaussian1D


class CubicIceModel:
    def __init__(self, amplitude, mean, stddev):
        self.fitter = LevMarLSQFitter()
        self.fitted_model = None
        self.model = self.__init_model__(amplitude, mean, stddev)

    def __init_model__(self, amplitude, mean, stddev):
        return Gaussian1D(amplitude=amplitude, mean=mean, stddev=stddev, name='cubic')

    def fit(self, x_data, y_data):
        self.fitted_model = self.fitter(self.model, x_data, y_data)
        self.__get_models__()

    def __get_models__(self):
        return self.fitted_model


class HexIceModel(CubicIceModel):
    def __init__(self,
                 amplitudes: tuple[int, int, int],
                 means: tuple[int, int, int],
                 stddevs: tuple[int, int, int]):
        super().__init__(amplitudes, means, stddevs)

    def __init_model__(self, amplitude: tuple[int, int, int],
                       mean: tuple[int, int, int],
                       stddev: tuple[int, int, int]):
        self.hex_1 = Gaussian1D(amplitude=amplitude[0], mean=mean[0], stddev=stddev[0],
                                name='hex_1')
        self.hex_2 = Gaussian1D(amplitude=amplitude[1], mean=mean[1], stddev=stddev[1],
                                name='hex_2')
        self.hex_3 = Gaussian1D(amplitude=amplitude[2], mean=mean[2], stddev=stddev[2],
                                name='hex_3')
        return self.hex_1 + self.hex_2 + self.hex_3

    def __get_models__(self):
        self.hex_1 = self.fitted_model['hex_1']
        self.hex_2 = self.fitted_model['hex_2']
        self.hex_3 = self.fitted_model['hex_3']


def gaussian_fitting_testing():
    m = Gaussian1D(amplitude=10, mean=20, stddev=5) + Gaussian1D(amplitude=10, mean=50, stddev=5) + Gaussian1D(
        amplitude=10, mean=80, stddev=5)
    x = np.linspace(0, 100, 2000)
    data = m(x)
    data = data + np.sqrt(data) * (np.random.random(x.size) - 0.5)
    # data -= data.min()
    plt.plot(x, data)

    HexIce = HexIceModel(amplitudes=[15, 8, 12], means=[15, 52, 79], stddevs=[4, 5, 6])
    HexIce.fit(x, data)

    print(f'Mean: {HexIce.hex_1.mean.value}, Amplitude: {HexIce.hex_1.amplitude.value}, FWHM: {HexIce.hex_1.fwhm}')
    print(f'Mean: {HexIce.hex_2.mean.value}, Amplitude: {HexIce.hex_2.amplitude.value}, FWHM: {HexIce.hex_2.fwhm}')
    print(f'Mean: {HexIce.hex_3.mean.value}, Amplitude: {HexIce.hex_3.amplitude.value}, FWHM: {HexIce.hex_3.fwhm}')

    plt.plot(x, HexIce.fitted_model(x))
    plt.show()


if __name__ == '__main__':
    gaussian_fitting_testing()
