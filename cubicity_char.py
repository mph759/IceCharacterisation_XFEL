"""
Cubicity characterisation of ice diffraction peaks from XFEL experiments
Written for use at PAL-XFEL experiment 2023-2nd-XSS-040
Created on 2024-03-26 by Michael Hassett
"""


def normalise_peaks(peaks: list[float, float, float]) -> list[float, float, float]:
    """
    Normalise the peaks of known hexagonal ice diffraction
    :param peaks: Hexagonal ice diffraction peaks
    :return: List of normalised peaks
    """
    return [peak/peaks[0] for peak in peaks]


def cubic_fraction(normalised_peak: float, peaks: list[float, float, float]):
    """
    Calculate the cubic fraction from the normalised hexagonal ice peaks
    :param normalised_peak: Second hexagonal ice diffraction peak
    :param peaks: Peaks found in uncharacterised ice diffraction
    :return:
    """
    return peaks[1] - normalised_peak * peaks[0]


def cubicity(normalised_peak: float, peaks: list[float, float, float]) -> float:
    """
    Cubicity characterisation of ice crystal from X-ray diffraction peaks
    :param normalised_peak: Normalised second hexagonal peak value (the peak shared with cubic)
    :param peaks: Peaks from uncharacterised ice diffraction data
    :return: Percentage of sample which is cubic
    """
    hex_fraction = peaks[0]
    cub_fraction = cubic_fraction(normalised_peak, peaks)
    return cub_fraction / (cub_fraction + hex_fraction)


def cubicity_testing():
    # Intensities of known 100% hexagonal ice diifraction peaks
    hex_peak1 = 17.491
    hex_peak2 = 9.316
    hex_peak3 = 10.188

    norm_peaks = normalise_peaks([hex_peak1, hex_peak2, hex_peak3])
    print(norm_peaks)

    # Intensities of uncharacterised Ice peaks
    peaks_2um = [3.374151, 16.07073, 1.04]
    cubicity_2um = cubicity(norm_peaks[1], peaks_2um)
    print(cubicity_2um)


if __name__ == '__main__':
    cubicity_testing()
