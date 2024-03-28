"""
Ice characterisation (cubicity and domain size) from XFEL diffraction data
Written for use at PAL-XFEL experiment 2023-2nd-XSS-040
Created on 2024-03-26 by Michael Hassett
"""

import numpy as np
import matplotlib.pyplot as plt
from ice_q_predict import IcePeakPrediction
from peak_fitting import CubicIceFitting, HexIceFitting
from cubicity_char import normalise_peaks, cubicity
from domain_size_char import domain_size

if __name__ == '__main__':
    pass