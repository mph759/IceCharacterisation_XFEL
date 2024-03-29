"""
Ingesting and pre-processing data from PAL-XFEL experiments
Written for use at PAL-XFEL experiment 2023-2nd-XSS-040
Created on 2024-03-29 by Michael Hassett
"""
import numpy as np
import h5py
from pathlib import Path
import pandas as pd

def read_data(data_path):
    raw_data = h5py.File(data_path, 'r')


if __name__ == '__main__':
    test_path = Path().cwd() / 'test' / '1h.h5'
    x = h5py.File(test_path, 'r')

    print(x.keys())
