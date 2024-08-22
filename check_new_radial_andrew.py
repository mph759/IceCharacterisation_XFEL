import numpy as np
import matplotlib.pyplot as plt
import h5py

#file = "/Volumes/BackUp19/Data/gspark_snu PAL-XFEL data/radial_averages/day4_run1_shot1_00001_DIR_mra.h5"
file = "/Users/andrewmartin/cloudstor/Work/Research/LF/Experiment/XFEL/PAL/2024apr/radial_averages/day4_run1_shot1_00001_DIR_mra.h5"

intensity_list = []
q_list = []

with h5py.File(file) as hf:

    scanIds = list(hf.keys())
    hfdict = {}
    for scanId in scanIds:
        #print("scanId", scanId)
        pulseIds = list(hf[scanId].keys())
        #print("pulseIds", pulseIds)
        hfdict[scanId] = pulseIds
    """ 
    s = scanIds[0]
    p = hfdict[s][0]
    q = hf[s][p]["q"][:]
    intensity = hf[s][p]["intensity"][:]
    #print(intensity.shape, np.max(intensity), np.min(intensity))
    plt.plot(q, intensity)
    """

    for s in hf.keys():
        for p in hf[s].keys():
            q = hf[s][p]["q"][:]
            intensity = hf[s][p]["intensity"][:]              
            intensity_list.append(intensity)
            q_list.append(q)

intensity_list = np.array(intensity_list)
q_list = np.array(q_list)
print("Number of pulses read: ", len(q_list))

nc = 20
for i in range(intensity_list.shape[0]-nc):
    intensity_list[i] = np.average(intensity_list[i:i+nc,:],0)



peakpos = [1.646,1.75, 1.858, 2.83]
w = 0.02

peak_maxes = np.zeros((len(q_list),len(peakpos)))
for i, q, intensity in zip(range(q_list.shape[0]),q_list, intensity_list):
    for ip in range(len(peakpos)):
        peak_maxes[i,ip] = np.max(intensity[ ((peakpos[ip]-w)<q)*(q<(peakpos[ip]+w))])

plt.figure()
plt.plot(peak_maxes[:,0])
plt.plot(peak_maxes[:,1])
plt.plot(peak_maxes[:,2])
plt.plot(peak_maxes[:,3])

#HACK
#plt.plot(peak_maxes[:,2]*8)

plt.figure()
il = [0, 400, 800, 1200, 1900]
nav = 10
for j in il:
    plt.plot(q_list[0,:], np.average(intensity_list[j:j+nav,:],0))

plt.figure()
#plt.plot( peak_maxes[:,0]/7.41 )

offset = 0.0
cubic_contribution =  offset + peak_maxes[:,1] - peak_maxes[:,0]/7.41 
cubic_contribution_p5 = offset +  peak_maxes[:,3] - peak_maxes[:,0]*0.7 
cubic_contribution_p3 =  offset + peak_maxes[:,2] - peak_maxes[:,0]*0.135 
plt.plot( cubic_contribution )
plt.plot( cubic_contribution_p5 )
plt.plot( cubic_contribution_p3 )
plt.plot( cubic_contribution_p5/cubic_contribution )

plt.figure()
plt.plot( cubic_contribution )
plt.plot( peak_maxes[:,0]/7.41)
plt.plot( peak_maxes[:,1])
plt.title( "Contributions to the 2nd peak")

plt.figure()

plt.title( "hex contributions of 2nd and 3rd peak")
p, = plt.plot( peak_maxes[:,0]/7.41)
p2, = plt.plot( peak_maxes[:,2] )
p3, = plt.plot(  (peak_maxes[:,0]/7.41) / peak_maxes[:,2] )
plt.legend([p,p2,p3],["2nd predicted", "3rd peak", "ratio"] )

#fitline = 0.4/(0.00075*(np.arange(q_list.shape[0])+300)) + 0.03
plt.figure()
plt.plot( cubic_contribution / (peak_maxes[:,0]+cubic_contribution))
plt.plot( peak_maxes[:,1] / (peak_maxes[:,0]+peak_maxes[:,1]))
plt.ylabel("cubicity")
plt.xlabel("pulse number")




plt.figure()
plt.plot( peak_maxes[:,1]/peak_maxes[:,0])
#plt.plot( fitline )
plt.plot( peak_maxes[:,2]/peak_maxes[:,0])

# get sigma  and peakvsigma values
sigmasamp = np.load("./pofigs/220824_sigma.npy")
peakvsig = np.load("./pofigs/220824_peakratios.npy")

def peakval_to_sigma( peakval, peakindex, sigmasamp, peakvsig):
    tmp = np.abs( peakvsig[:,peakindex]-peakval)
    index = np.where(tmp == np.min(tmp))
    sigma = sigmasamp[index]
    return sigma

def sigma_to_peakval( sigmaval, peakindex, sigmasamp, peakvsig):
    tmp = np.abs( sigmasamp -sigmaval)
    index = np.where(tmp == np.min(tmp))
    peakval = peakvsig[index, peakindex]
    return peakval


print('Test peakval to sigma', peakval_to_sigma( 0.3, 2, sigmasamp, peakvsig))

sigmavals_from_data = np.zeros(peak_maxes.shape[0])
peak2hex_ratio = np.zeros(peak_maxes.shape[0])
for i in range(peak_maxes.shape[0]):
    sigmavals_from_data[i] = peakval_to_sigma( peak_maxes[i,2]/peak_maxes[i,0], 2, sigmasamp, peakvsig)
    peak2hex_ratio[i] = sigma_to_peakval( sigmavals_from_data[i], 1, sigmasamp, peakvsig) 
plt.figure()
plt.plot( sigmavals_from_data )
plt.ylabel( "sigma") 
plt.xlabel( "pulse number")

plt.figure()
plt.plot( peak2hex_ratio)
plt.title("peak 2 ratio predicted from sigma")
plt.figure()
plt.plot( peak2hex_ratio*peak_maxes[:,0])
plt.plot( peak_maxes[:,1])

#corrected 
corrected_secondpeak =  peak_maxes[:,1] -  peak2hex_ratio*peak_maxes[:,0]
plt.plot( corrected_secondpeak)

plt.figure()
plt.plot( sigmasamp, peakvsig[:,1])
plt.plot( sigmasamp, peakvsig[:,2])
plt.xlabel("sigma")
plt.ylabel("peak ratio")

# some other test
hex2_contribution_predicted = (peak_maxes[:,1] - 0.68)/peak_maxes[:,0]
plt.figure()
plt.plot(hex2_contribution_predicted)

sigmaval_blah = peakval_to_sigma( 0.165, 1, sigmasamp, peakvsig)
print( "predicted asymptotic sigma value from blah", sigmaval_blah)

plt.draw()
plt.show()
