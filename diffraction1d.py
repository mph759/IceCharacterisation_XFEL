"""
Radial integration of diffraction peaks from XFEL experiments
Written for use at PAL-XFEL experiment 2023-2nd-XSS-040
Created on 2024-03-26 by Michael Hassett
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pyFAI
from pyFAI.detectors import Detector
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.geometry import Geometry
import os


def radial_integration(frame: np.ndarray):
    """
    Perform azimuthal integration of frame array
    :param frame: numpy array containing 2D intensity
    :param unit: Unit used in the radial integration
    :return: two-col array of q & intensity.
    """
    # print("Debug - ", self.detector_dist, self.pixel_size, self.wavelength)

    setting, mask, bkg = load_onai_config()

    x_centre, y_centre = [1458.937, 1427.764]
    if 'rayonix_lx255-hs' == setting['detector']:
        detector = Detector(pixel1=0.255 / 5760, pixel2=0.255 / 5760, max_shape=(5760, 1920))
    else:
        detector = pyFAI.detector_factory(setting['detector'])
    detector.set_binning((setting['binning'], setting['binning']))
    poni1 = (setting['dim1'] - y_centre) * detector.pixel1
    poni2 = x_centre * detector.pixel2
    ai = AzimuthalIntegrator(dist=setting['distance'],
                             poni1=poni1, poni2=poni2,
                             rot1=np.deg2rad(setting['rot1']),
                             rot2=np.deg2rad(setting['rot2']),
                             rot3=np.deg2rad(setting['rot3']),
                             detector=detector)

    if 'rayonix' == setting['detector'][:7]:
        mu_times_l = 1 / 9.72584 * 40  # from cxro site about Gd2O2S at 12.7 keV
        pyFAI_geom = Geometry(dist=setting['distance'],
                              poni1=poni1, poni2=poni2,
                              rot1=np.deg2rad(setting['rot1']),
                              rot2=np.deg2rad(setting['rot2']),
                              rot3=np.deg2rad(setting['rot3']),
                              detector=detector)
        phos_cor = pyFAI_geom.calc_transmission(np.exp(-mu_times_l))
        bkg = np.divide(bkg, phos_cor)

    # run once to reduce the initial jitter
    ai_res = ai.integrate1d(frame, setting['n_step'], mask=mask, dark=bkg,
                            polarization_factor=setting['polarization_factor'], unit=setting['unit'])
    return ai_res


def load_onai_config(filename=None, det_type='rayonix_lx255-hs'):
    if 'rayonix_lx255-hs' == det_type:
        f_nslow = 5760
        f_nfast = 1920
    elif 'rayonix_mx225hs' == det_type:
        f_nslow = 5760
        f_nfast = 5760
    else:
        print('load_onai_config - Error')
        print('Unknown Detector Type : %s' % (det_type))
        return [None, None, None]

    default_setting = {}
    default_setting['detector'] = det_type
    default_setting['binning'] = 4
    default_setting['dim1'] = int(f_nslow / default_setting['binning'])  # nslow
    default_setting['dim2'] = int(f_nfast / default_setting['binning'])  # nfast
    default_setting['distance'] = 0.229095
    default_setting['x_cen'] = default_setting['dim2'] * 0.5  # 722.985
    default_setting['y_cen'] = default_setting['dim1'] * 0.5  # 724.675 #default_setting['dim1'] - 724.675
    default_setting['rot1'] = 0.0  # deg
    default_setting['rot2'] = 0.0  # deg
    default_setting['rot3'] = 0.0  # deg
    default_setting['polarization_factor'] = 0.996
    default_setting['n_step'] = 512
    default_setting['unit'] = '2th_deg'

    default_setting['mask_version'] = 'nothing'
    # 1 - masked pixel, 0 - valid pixel
    default_mask = np.zeros([default_setting['dim1'], default_setting['dim2']], dtype=np.int8)

    # default_setting['background_level'] = 9.8
    default_setting['bkg_version'] = 'constant_9.8'
    default_bkg = np.ones([default_setting['dim1'], default_setting['dim2']], dtype=np.float32) * 9.8

    if filename is None:
        print('Load Default Configuration')
        return default_setting, default_mask, default_bkg


if __name__ == '__main__':
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
