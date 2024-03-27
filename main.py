"""
Ice characterisation (cubicity and domain size) from XFEL diffraction data
Written for use at PAL-XFEL experiment 2023-2nd-XSS-040
Created on 2024-03-26 by Michael Hassett
"""

import numpy as np
import matplotlib.pyplot as plt
from peak_fitting import CubicIceModel, HexIceModel
from cubicity import normalise_peaks, cubicity
from domain_size import domain_size

if __name__ == '__main__':
    pass