"""
Prediction of q vector values of ice peaks in hexagonal and cubic ice from XFEL diffraction data
Written for use at PAL-XFEL experiment 2023-2nd-XSS-040
Created on 2024-03-26 by Michael Hassett
"""

import numpy as np
import matplotlib.pyplot as plt


class IcePeakPrediction:
    def __init__(self, wavelength: float, x_param: str = 'q'):
        """
        Calculator of the q vector values of ice peaks in hexagonal and cubic ice from XFEL diffraction data
        :param wavelength: Wavelength used in the diffraction experiment (in metres)
        :param x_param: X-axis parameter (Default q vector), to which the peaks will be applied. Should be q or 2theta
        """
        self.__wavelength__ = wavelength
        if x_param not in ['q', '2theta']:
            raise ValueError('x_param must be either q or 2theta')
        if x_param == 'q':
            self.__x_param__ = 'q'
        if x_param == '2theta':
            self.__x_param__ = '2theta'
        self.__hex_peaks__ = self.hex_ice_peaks()
        self.__cubic_peak__ = self.cubic_ice_peak()

    @property
    def wavelength(self) -> float:
        return self.__wavelength__

    @property
    def x_param(self) -> str:
        return self.__x_param__

    @property
    def hex_peaks(self) -> list[float, float, float]:
        return self.__hex_peaks__

    @property
    def cubic_peak(self) -> list[float]:
        return self.__cubic_peak__

    @property
    def peaks(self) -> dict:
        return {'hex': self.hex_peaks, 'cubic': self.cubic_peak}

    def hex_ice_peaks(self) -> list[float, float, float]:
        raise NotImplemented

    def cubic_ice_peak(self) -> list[float]:
        raise NotImplemented

    def add2plot(self, ax: plt.axes) -> None:
        """
        Add a line showing where the peaks should be, drawn on the axes provided
        :param ax: Matplotlib Axes object to be drawn on
        :return: None
        """
        ax.plot(self.hex_peaks, 0)
        ax.plot(self.cubic_peak, 0)


if __name__ == '__main__':
    pass
