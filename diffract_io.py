"""
Ingesting and pre-processing data from PAL-XFEL experiments
Written for use at PAL-XFEL experiment 2023-2nd-XSS-040
Created on 2024-03-29 by Michael Hassett (Edited from original code by Sebastian Cardoch)
"""
import numpy as np
import h5py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors


class diffraction_h5:
    def __init__(self, data_path, q_selection):
        self.__intensity__, self.__rad_intensity__, self.__sx__, self.__sy__, self.__q__ = (
            self.read_data(data_path))
        self.q_selection = np.sort(q_selection)

    @property
    def intensity(self): return self.__intensity__

    @property
    def rad_intensity(self): return self.__rad_intensity__

    @property
    def sx(self): return self.__sx__

    @property
    def sy(self): return self.__sy__

    @property
    def q(self): return self.__q__

    def read_data(self, data_path):
        with h5py.File(data_path, 'r') as f:
            intensity = f["intensity"][()]
            rad_intensity = f["radintensity"][()]
            sx = f["sx"][()]
            sy = f["sy"][()]
            sr = f["sr"][()]
        return intensity, rad_intensity, sx, sy, sr

    def plot_intensity(self, fig: plt.figure = None, ax: plt.axes = None):
        if ax is None:
            if fig is not None:
                raise ValueError("Must provide an axes object if providing figure object")
            else:
                fig, ax = plt.subplots()
        ax.imshow(self.intensity, extent=[self.sx[0, -1], self.sx[0, 0], self.sy[-1, 0], self.sy[0, 0]],
                  norm=colors.LogNorm(), aspect="equal")
        ax.set_title("diffraction pattern")
        return fig, ax

    def plot_rad_intensity(self, fig: plt.figure = None, ax: plt.axes = None):
        if ax is None:
            if fig is not None:
                raise ValueError("Must provide an axes object if providing figure object")
            else:
                fig, ax = plt.subplots()
        ax.plot(self.q[(self.q_selection[0] < self.q) & (self.q < self.q_selection[1])],
                self.rad_intensity[(self.q_selection[0] < self.q) & (self.q < self.q_selection[1])])
        ax.set_xlabel("momentum transfer [A$^{-1}$]")
        ax.set_title("radial average")
        return fig, ax

    def plot_together(self, title: str=None):
        fig, ax = plt.subplots(1, 2, figsize=(8, 4), layout="tight")
        self.plot_intensity(fig, ax[0])
        self.plot_rad_intensity(fig, ax[1])
        if title is not None:
            fig.suptitle(title)
        return fig, ax


def load_test_data():
    test_path = Path().cwd() / 'test'
    test_files = {'cubic ice': '1c.h5',
                  'hexagonal ice': '1h.h5',
                  'hexagonal + cubic ice': 'hc.h5'}
    q_select = [1.5, 2]

    for dataset, data_path_subdir in test_files.items():
        data_dir = test_path / data_path_subdir
        diff_pattern = diffraction_h5(data_dir, q_select)
        diff_pattern.plot_together(dataset)


if __name__ == '__main__':
    load_test_data()
    plt.show()
