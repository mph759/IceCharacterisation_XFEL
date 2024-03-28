"""
Cubicity characterisation of ice diffraction peaks from XFEL experiments
Written for use at PAL-XFEL experiment 2023-2nd-XSS-040
Created on 2024-03-26 by Michael Hassett
"""
from typing import List, Any

import numpy as np


def normalise_peaks(peaks: list[float]) -> list[float]:
    """
    Normalise the peaks of known hexagonal ice diffraction
    :param peaks: Hexagonal ice diffraction peaks
    :return: List of normalised peaks
    """
    normal_peak = np.max(peaks)
    return [peak/normal_peak for peak in peaks]


def cubic_fraction(normalised_peak: float, peaks: list[float]):
    """
    Calculate the cubic fraction from the normalised hexagonal ice peaks
    :param normalised_peak: Second hexagonal ice diffraction peak
    :param peaks: Peaks found in uncharacterised ice diffraction
    :return:
    """
    return peaks[1] - normalised_peak * peaks[0]


def cubicity(normalised_peak: float, peaks: list[float]) -> float:
    """
    Cubicity characterisation of ice crystal from X-ray diffraction peaks
    :param normalised_peak: Normalised second hexagonal peak value (the peak shared with cubic)
    :param peaks: Peaks from uncharacterised ice diffraction data
    :return: Percentage of sample which is cubic
    """
    hex_fraction = peaks[0]
    cub_fraction = cubic_fraction(normalised_peak, peaks)
    cubicity_percent = cub_fraction / (cub_fraction + hex_fraction)
    return cubicity_percent


def cubicity_testing():
    # Intensities of known 100% hexagonal ice diffraction peaks
    hex_peaks = [17.491, 9.316, 10.188]

    norm_peaks = normalise_peaks(hex_peaks)
    print(norm_peaks)

    # Intensities of uncharacterised Ice peaks
    ice_2um_peaks = [3.374151, 16.07073, 1.04]
    ice_10um_peaks = [67.89297, 30.95, 8.3]
    ice_50um_peaks = [172.57, 28.44, 20.47]
    cubicity_2um = cubicity(norm_peaks[1], ice_2um_peaks)
    cubicity_10um = cubicity(norm_peaks[1], ice_10um_peaks)
    cubicity_50um = cubicity(norm_peaks[1], ice_50um_peaks)
    print(f'2 micron thick ice: {cubicity_2um}')
    print(f'10 micron thick ice: {cubicity_10um}')
    print(f'50 micron thick ice: {cubicity_50um}')


if __name__ == '__main__':
    cubicity_testing()
