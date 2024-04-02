"""
Cubicity characterisation of ice diffraction peaks from XFEL experiments
Written for use at PAL-XFEL experiment 2023-2nd-XSS-040
Created on 2024-03-26 by Michael Hassett
"""
from typing import List, Any

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
# testing
from peak_fitting import sim_data_fitting_testing


def normalise_peaks(peaks: list):
    """
    Normalise the peaks of known hexagonal ice diffraction
    :param peaks: Hexagonal ice diffraction peaks
    :return: List of normalised peaks
    """
    normal_peak = np.max(peaks)
    return [peak / normal_peak for peak in peaks]


def cubic_fraction(normalised_peak: float, peaks: list):
    """
    Calculate the cubic fraction from the normalised hexagonal ice peaks
    :param normalised_peak: Second hexagonal ice diffraction peak
    :param peaks: Peaks found in uncharacterised ice diffraction
    :return:
    """
    return peaks[1] - normalised_peak * peaks[0]


def cubicity(hex_peaks: list, peaks: list) -> float:
    """
    Cubicity characterisation of ice crystal from X-ray diffraction peaks
    :param hex_peaks: Known hexagonal ice peak values
    :param peaks: Peaks from uncharacterised ice diffraction data
    :return: Percentage of sample which is cubic
    """
    hex_fraction = peaks[0]
    normalised_peak = normalise_peaks(hex_peaks)
    cub_fraction = cubic_fraction(normalised_peak[1], peaks)
    cubicity_percent = cub_fraction / (cub_fraction + hex_fraction)
    return cubicity_percent


def cubicity_testing():
    # Intensities of known 100% hexagonal ice diffraction peaks
    hex_peaks = [17.491, 9.316, 10.188]

    # Intensities of uncharacterised Ice peaks
    ice_2um_peaks = [3.374151, 16.07073, 1.04]
    ice_10um_peaks = [67.89297, 30.95, 8.3]
    ice_50um_peaks = [172.57, 28.44, 20.47]
    cubicity_2um = cubicity(hex_peaks, ice_2um_peaks)
    cubicity_10um = cubicity(hex_peaks, ice_10um_peaks)
    cubicity_50um = cubicity(hex_peaks, ice_50um_peaks)
    print(f'2 micron thick ice: {cubicity_2um}')
    print(f'10 micron thick ice: {cubicity_10um}')
    print(f'50 micron thick ice: {cubicity_50um}')


def cubicity_sim_data_test():
    test_path = Path().cwd() / 'test'
    hex_ice_file = '1h.h5'

    data_path = test_path / hex_ice_file
    hex_ice_model = sim_data_fitting_testing(data_path, 'hexagonal ice',
                                             amplitude=[6.5e-5, 1.3e-4, 2.9e-5],
                                             mean=[1.6, 1.7, 1.8],
                                             stddev=[0.01, 0.01, 0.01])
    hex_ice_model_peaks = [hex_ice_model.peak1.amplitude.value,
                           hex_ice_model.peak2.amplitude.value,
                           hex_ice_model.peak3.amplitude.value]
    hex_ice_model_norm_peaks = normalise_peaks(hex_ice_model_peaks)

    mixed_ice_file = 'hc.h5'
    data_path = test_path / mixed_ice_file
    mixed_ice_model = sim_data_fitting_testing(data_path, 'hexagonal + cubic ice',
                                               amplitude=[6.3e-5, 1.3e-4, 2.61e-5],
                                               mean=[1.6, 1.7, 1.8],
                                               stddev=[0.01, 0.01, 0.005])
    print(mixed_ice_model.fitter.fit_info['message'])
    mixed_ice_model_peaks = [mixed_ice_model.peak1.amplitude.value,
                             mixed_ice_model.peak2.amplitude.value,
                             mixed_ice_model.peak3.amplitude.value]
    mixed_ice_cubicity = cubicity(hex_ice_model_peaks, mixed_ice_model_peaks)
    print(f'Cubicity: {mixed_ice_cubicity}')
    plt.show()


if __name__ == '__main__':
    cubicity_testing()
    cubicity_sim_data_test()
