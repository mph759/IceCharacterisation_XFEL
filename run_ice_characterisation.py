"""
Runner script for characterisation of ice diffraction peaks from XFEL experiments
Written for use at PAL-XFEL experiment 2023-2nd-XSS-040
Created on 2024-03-30 by Michael Hassett
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

from peak_fitting import CubicIceFitting, HexIceFitting
from ice_peak_predict import IcePeakPrediction
from cubicity_char import normalise_peaks, cubicity
from domain_size_char import domain_size
import paltools


if __name__ == '__main__':
    experiment_id = '2023-2nd-XSS-040'
    detector_dist = 0.226   # metres
    photon_energy = 15e3    # eV
    wavelength = paltools.energy2wavelength(photon_energy)  # metres

    root_path = Path.cwd()

    run_id = "scratch-20240331T002819Z-001\scratch"
    run = paltools.run(experiment_id, root_path, run_id)
    scan_id = 1

    # tiff_path = run_id / Path('day4_run1_Ice_1_00001_avr.tiff')
    # tiff_array = np.array(Image.open(tiff_path))
    # plt.imshow(tiff_array)
    # r, radial_average = paltools.radial_average(tiff_array, 1)
    plt.figure()
    # plt.plot(r, radial_average)
    ttheta = paltools.bins2twotheta(r, detector_dist)
    q_vector = paltools.twothetaq(ttheta, wavelength) / 1e9
    q_selection = (q_vector > 4)
    plt.plot(q_vector[q_selection], abs(radial_average[q_selection]))

    # plt.plot(q_vector, radial_average)

    plt.xlabel("q \ nm$^{-1}$")
    plt.ylabel("Intensity")

    hex_ice_peaks_q = IcePeakPrediction(wavelength, 'q').peaks['hex'] / 1e9
    ice_peaks_ttheta = IcePeakPrediction(wavelength, '2theta').peaks
    print(hex_ice_peaks_q)

    '''
    ttheta, intensity = run.getRadialAverage(scan_id, run.getPulseIds(scan_id)[0])
    q_vector = paltools.twothetaq(ttheta, 15e3) / 1e10
    plt.plot(q_vector, intensity)
    
    amp = [84, 35, 23]
    mean = [2, 2.72, 3.35]
    stddev = [0.05, 0.01, 0.01]
    
    three_peaks = HexIceFitting(amplitude=amp, mean=mean, stddev=stddev, name=['p1', 'p2', 'p3'])
    three_peaks.fit(q_vector, intensity)
    for i in range(0, 3):
        peak = three_peaks.model_data(i)
        for key, item in peak.items():
            print(f'{key}: {item}')
        print('\n')
    
    plt.plot(q_vector, three_peaks.model(q_vector))
    '''
    plt.show()