"""
Prediction of q vector values of ice peaks in hexagonal and cubic ice from XFEL diffraction data
Written for use at PAL-XFEL experiment 2023-2nd-XSS-040
Created on 2024-03-26 by Michael Hassett
"""
import matplotlib.pyplot as plt
import numpy as np

import paltools


def peak_predict_q(d_spacings: np.ndarray):
    """
    Predict the peak q vector values of a crystal with specified d spacing(s)
    :param d_spacings: Lattice spacings of crystal in metres
    :return: peak_locs: peak positions in q
    """
    # peak_locs = sorted(np.round(np.divide(num_pixels, d_spacings)))
    peak_locs = np.sort(2 * np.pi / d_spacings)
    return peak_locs


def peak_predict_2theta(wavelength: float, d_spacings: np.ndarray):
    """
    Predict the peak 2theta values of a crystal with specified d spacing(s), and experimental wavelength
    :param wavelength: Experimental diffraction wavelength in metres
    :param d_spacings: Lattice spacings of crystal in metres
    :return: peak_locs: peak positions in 2theta
    """
    remainder, ratio = np.divmod(wavelength/d_spacings, np.pi)
    peak_locs = np.arcsin(remainder)
    peak_locs = peak_locs * ratio
    return peak_locs


class IcePeakPrediction:
    def __init__(self, wavelength: float, x_param: str = 'q'):
        """
        Calculator of the q vector values of ice peaks in hexagonal and cubic ice from XFEL diffraction data
        :param wavelength: Wavelength used in the diffraction experiment (in metres)
        :param x_param: X-axis parameter (Default q vector), to which the peaks will be applied. Should be q or 2theta
        """
        self.__wavelength__ = wavelength
        self.x_param = x_param
        self.__hex_d_spacing__ = np.array([0.78228388e-9, 0.73535726e-9, 0.903615185e-9])
        self.__cubic_d_spacing__ = np.array([0.63818213e-9])
        self.hex_ice_peaks()
        self.cubic_ice_peak()

    @property
    def wavelength(self) -> float:
        return self.__wavelength__

    @property
    def x_param(self) -> str:
        return self.__x_param__

    @x_param.setter
    def x_param(self, value):
        if value not in ['q', '2theta']:
            raise ValueError('x_param must be either q or 2theta')
        if value == 'q':
            self.__x_param__ = 'q'
        if value == '2theta':
            self.__x_param__ = '2theta'

    @property
    def hex_peaks(self) -> list:
        return self.__hex_peaks__

    @property
    def cubic_peak(self) -> list:
        return self.__cubic_peak__

    @property
    def hex_d_spacing(self) -> list:
        return self.__hex_d_spacing__

    @property
    def cubic_d_spacing(self) -> list:
        return self.__cubic_d_spacing__

    @property
    def peaks(self) -> dict:
        return {'hex': self.hex_peaks, 'cubic': self.cubic_peak}

    def hex_ice_peaks(self) -> list:
        self.__hex_peaks__ = self.__determine_peak_locs__(self.hex_d_spacing)

    def cubic_ice_peak(self) -> list:
        self.__cubic_peak__ = self.__determine_peak_locs__(self.cubic_d_spacing)

    def __determine_peak_locs__(self, d_spacing):
        if self.__x_param__ == 'q':
            peak_locs = peak_predict_q(d_spacing)
        elif self.__x_param__ == '2theta':
            peak_locs = peak_predict_2theta(self.wavelength, d_spacing)
        else:
            raise ValueError('x_param must be either q or 2theta')
        return peak_locs

    def add2plot(self, ax: plt.axes) -> None:
        """
        Add a line showing where the peaks should be, drawn on the axes provided
        :param ax: Matplotlib Axes object to be drawn on
        :return: None
        """
        ax.plot(self.hex_peaks, 0)
        ax.plot(self.cubic_peak, 0)


def ice_peak_prediction_testing():
    # wavelength = 0.84e-9  # metres
    energy = 15e3
    wavelength = paltools.energy2wavelength(energy)
    print(wavelength)
    ice = IcePeakPrediction(wavelength=wavelength, x_param='2theta')
    print(ice.hex_peaks)
    print(ice.cubic_peak)


if __name__ == '__main__':
    ice_peak_prediction_testing()
