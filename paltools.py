import h5py

import numpy
ROOT = "/pal/home/gspark_snu/ctbas/ue_240330_FXL/scan/"


    


class run:
    def __init__(self,runname):
        self.runname = ROOT + runname + "/"
        self.pulseinfo_filename = self.runname + "/pulseInfo/"
        self.twotheta_filename = self.runname + "/eh1rayMXAI_tth/"
        self.intensity_filename = self.runname + "/eh1rayMXAI_int/"

    def getPulseIds(self,scanId):
        filename = self.pulseinfo_filename + "001_001_%03d.h5"%scanId
        with h5py.File(filename) as f:
            pulseIds = list(f.keys())
        return pulseIds


    def getRadialAverage(self,scanId,pulseId):
        scan_name = "001_001_%03d.h5"%scanId
        twotheta_filename =self.twotheta_filename + scan_name
        intensity_filename =self.intensity_filename + scan_name
        xvar = run.__getRadial(twotheta_filename,pulseId)
        yvar = run.__getRadial(intensity_filename,pulseId)
        return xvar,yvar
    
    @staticmethod
    def __getRadial(filename,pulseId):
        with h5py.File(filename) as f:
            radial = f[pulseId][()]
        return radial


def twoThetaToq(twoTheta,photon_energy):
    wavelength = energy2wavelength(photon_energy)
    return (4*numpy.pi/wavelength) * numpy.sin(numpy.deg2rad(twoTheta)/2)

def energy2wavelength(photon_energy):
    c = 299792458#[m/s]
    h =6.582119569e-16 #[eV*s]
    return c * h / photon_energy