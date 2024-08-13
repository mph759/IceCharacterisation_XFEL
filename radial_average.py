import os
import sys
sys.path.append("../../../Downloads")

import pyFAI
import numpy
import pyFAI.detectors
from paltools import Experiment, Run
import h5py
import glob
from pathlib import Path

ROOT = "/home/PAL/ue_240330_FXL/"
POLARIZATION_FACTOR = 0.996
UNIT = "q_A^-1"
PONIFILE = "/home/sebastian/projects/pal-xss-r004/data/detector-calibration/Aghb_2024_06_11.poni"


def run(ray_data, dark, mask, npt=5760 // 2, radial_range=None):
    ai = pyFAI.load(PONIFILE)
    mu_times_l = 1 / 9.72584 * 40  # from cxro site about Gd2O2S at 12.7 keV
    phos_cor = ai.calc_transmission(numpy.exp(-mu_times_l))
    dark = numpy.divide(dark, phos_cor)
    ray_data = numpy.divide(ray_data, phos_cor)
    r, I = ai.integrate1d_ng(ray_data, npt, mask=mask, dark=dark, polarization_factor=POLARIZATION_FACTOR, unit=UNIT, radial_range=radial_range, correctSolidAngle=True)
    return r, I


def radialavg(ai, ray_data, phos_cor, dark, mask, npt=5760 // 2, radial_range=None):
    ray_data = numpy.divide(ray_data, phos_cor)
    r, I = ai.integrate1d_ng(ray_data, npt, mask=mask, dark=dark, polarization_factor=POLARIZATION_FACTOR, unit=UNIT, radial_range=radial_range, correctSolidAngle=True)
    return r, I


def main(runname):
    root_path = Path('F:\gspark_snu PAL-XFEL data/ctbas/ue_240330_FXL/scan/')
    current_exp = Experiment(experiment_id="2023-2nd-XSS-040",
                             photon_energy=15e3,
                             detector_distance=0.321,
                             pixel_size=0.255 / 5760,
                             root_path=root_path)
    current_run = Run(current_exp, runname)
    mra_r_exists = Path(current_run.name + "/mra_r.h5").is_file()
    mra_int_exists = Path(current_run.name + "/mra_int.h5").is_file()
    # if mra_r_exists and mra_int_exists:
    # return None  # runs already processed

    if runname in ["day6_run1_shot1_00005_DIR", "day6_run1_shot1_00005_00001_DIR", "day6_run1_shot1_00004_DIR", "day6_run1_shot1_00002_DIR"]:
        return None  # no data saved on these runs

    print("processing %s" % current_run.name)

    pulseids = current_run.getPulseIds()
    # print(pulseids.shape)
    # print(run.getImagePulseIds().shape)
    try:
        pulseids = check_pulseids(pulseids, current_run.getImagePulseIds())
    except RuntimeError:
        pulseids = numpy.array(
            [
                "1712044015.0504627_50400",
                "1712044015.1504707_50436",
                "1712044015.2504785_50472",
                "1712044015.3504865_50508",
                "1712044015.4504945_50544",
                "1712044015.5505023_50580",
                "1712044015.6505103_50616",
                "1712044015.7506495_50652",
                "1712044015.8506572_50688",
                "1712044015.9506652_50724",
                "1712044016.0505939_50760",
                "1712044016.1506016_50796",
                "1712044016.2506096_50832",
                "1712044016.3507488_50868",
                "1712044016.4507565_50904",
                "1712044016.5507646_50940",
                "1712044016.6507726_50976",
                "1712044016.7507803_51012",
                "1712044016.8507884_51048",
                "1712044016.9507964_51084",
                "1712044017.0508559_51120",
                "1712044017.150864_51156",
                "1712044017.250872_51192",
                "1712044017.3508797_51228",
            ]
        )
        print("error getting image pulse ids")

    dark = paltools.loadDark()
    mask = paltools.loadMask()

    ai = pyFAI.load(PONIFILE)
    mu_times_l = 1 / 9.72584 * 40  # from cxro site about Gd2O2S at 12.7 keV
    phos_cor = ai.calc_transmission(numpy.exp(-mu_times_l))
    dark = numpy.divide(dark, phos_cor)

    q_array = []
    intensity_array = []
    for pulseids_chunk in chunker(pulseids, 1000):
        q_chunk, intensity_chunk = __process(current_run, pulseids_chunk, ai, phos_cor, dark, mask)
        q_array.append(q_chunk)
        intensity_array.append(intensity_chunk)

    q_array = numpy.vstack(q_array)
    intensity_array = numpy.vstack(intensity_array)
    __save(current_run, pulseids, q_array, intensity_array)
    return None


class PulseIdsMatchingError:
    pass


def check_pulseids(pd1, pd2):
    nmin = numpy.min([pd1.size, pd2.size])
    nmax = numpy.max([pd1.size, pd2.size])
    if pd1.size != pd2.size:
        print("run is incomplete. processing %d out of %d pulses" % (nmin, nmax))
    if any(pd1[:nmin] != pd2[:nmin]):
        raise PulseIdsMatchingError
    return pd1[:nmin]


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def __process(run, pulseids, ai, phos_cor, dark, mask):
    print("loading images...")
    images = run.getImage(pulseids)
    intensity_array = []
    q_array = []
    print("computing radial average...")
    for index in range(images.shape[0]):
        print("%d/%d" % (index, images.shape[0]), end="\r")
        q, rI = radialavg(ai, images[index], phos_cor, dark, mask)
        intensity_array.append(rI)
        q_array.append(q)
    return q_array, intensity_array


def __save(run, pulseids, q_array, intensity_array):
    print("saving radial averages...")
    with h5py.File(run.runname + "/mra_int.h5", "w") as f:
        for index in range(pulseids.size):
            f.create_dataset(pulseids[index], data=intensity_array[index])
    with h5py.File(run.runname + "/mra_r.h5", "w") as f:
        for index in range(pulseids.size):
            f.create_dataset(pulseids[index], data=q_array[index])
    return None


if __name__ == "__main__":
    runnames = glob.glob(ROOT + "scan/day*")
    runnames = [runname.split("/")[-1] for runname in runnames]
    for runname in runnames:
        main(runname)
