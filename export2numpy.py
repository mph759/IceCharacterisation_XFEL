from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import find_peaks
from copy import deepcopy

from paltools import Experiment, Run

def per_shot(run: Run, scan_id):
    z_dum = run.getPulseIds(scan_id)
    z_len = len(z_dum)
    x_dum, y_dum = run.getRadialAverage(scan_id, z_dum[0])
    y_len = len(y_dum)
    array_shape = [z_len, y_len]
    del x_dum, y_dum, z_dum
    data_array = np.zeros(array_shape)

    return data_array


if __name__ == '__main__':
    root_path = Path('F:\gspark_snu PAL-XFEL data/ctbas/ue_240330_FXL/scan/')
    current_exp = Experiment(experiment_id="2023-2nd-XSS-040",
                             photon_energy=15e3,
                             detector_distance=0.321,
                             pixel_size=0.255 / 5760,
                             root_path=root_path)
    run_root = 'day3_rus3_shot1'
    output_root = Path(f'output/{run_root}')
    runs = sorted(list(root_path.glob(f'{run_root}_*_DIR')))


    data_array = np.empty((len(runs)*300, 512))
    data_array[:] = np.nan
    print(data_array.shape)
    i_correction = 0
    for i, run in enumerate(runs):
        current_run = Run(current_exp, run.name)
        scan_id = 1
        #for scan_id in range(1, current_run.numscans+1):
        for pulse_num, pulse_id in enumerate(current_run.getPulseIds(scan_id)):
            try:
                xvar, yvar = current_run.getRadialAverage(scan_id, pulse_id)
                data_array[pulse_num+i*300] = yvar
            except KeyError as e:
                print(f'No data found for {pulse_id}, Scan {scan_id} in {current_run.name}')
                raise e

    np.save(output_root/f'{run_root}.npy', data_array)
