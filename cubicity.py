"""
Cubicity characterisation of ice diffraction peaks from XFEL experiments
Written for use at PAL-XFEL experiment 2023-2nd-XSS-040
Created on 2024-03-26 by Michael Hassett
"""


def normalise_peaks(*peaks):
    normalised_peaks = []
    peak0 = peaks[0]
    for i, peak in enumerate(peaks):
        normalised_peaks.append(peak / peak0)
    return normalised_peaks


def cubic_fraction(normalised_peak: float, peaks: list[float, float, float]):
    return peaks[1] - normalised_peak * peaks[0]


def cubicity(normalised_peak: float, *peaks):
    hex_fraction = peaks[0]
    cub_fraction = cubic_fraction(normalised_peak, peaks)
    return cub_fraction / (cub_fraction + hex_fraction)


if __name__ == '__main__':
    peak1 = 17.491
    peak2 = 9.316
    peak3 = 10.188

    peaks_2um = [3.374151, 16.07073, 1.04]
    norm_peaks = normalise_peaks(peak1, peak2, peak3)
    print(norm_peaks)
    cubicity_2um = cubicity(norm_peaks[1], *peaks_2um)
    print(cubicity_2um)
