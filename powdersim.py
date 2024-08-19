import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

def wk_sf( q, wk ):

    sf = wk[0]
    for i in range(1,6):
        sf += wk[i]*np.exp(-wk[5+i]*q*q/4.0)
    return sf    


outpath = "./pofigs/"
# HEX ICE
abc = np.array([4.506,4.506,7.346])
angles = np.array([90.0,90.0,120.0])
rad = np.deg2rad(angles)
vol = 129.170681


a = np.array( [abc[0],0,0] )
b = np.array( [abc[1]*np.cos(rad[2]),abc[1]*np.sin(rad[2]),0] )
c = np.array( [0,0,abc[2]] )

astar = np.cross(b,c)/vol
bstar = np.cross(c,a)/vol
cstar = np.cross(a,b)/vol

astar_mod = np.sqrt(np.sum(astar*astar))
cstar_mod = np.sqrt(np.sum(cstar*cstar))

#scattering factor (Waasmeier Kirfel parameters):
wk = [8, 2.960427,2.508818,0.637853,0.722838,1.142756,14.182259,5.936858,0.112726,34.958481,0.390240,0.027014,8.0]


# oxygen coordinates
xyz = np.array([
    [0.00000,   2.60154,   0.45398],
    [0.00000,   2.60154,   3.21902],
    [2.25300,   1.30077,   4.12698],
    [2.25300,   1.30077,   6.89202]
                ])


#some preferred orietnation stuff
erf_norm = erf(1)
print("erf norm", erf_norm)
sigma = 0.1

#check preferred orientation distributions
def calc_po_weight_ab_align( csth, sigma=0.5 ):
    po_norm   = 2*sigma*erf(1/(2*sigma))
    if csth <0:
        arg = csth + 1
    else:
        arg = csth -1
    po_w = (np.sqrt(2.0/np.pi))*np.exp(-(arg**2)/(2*sigma**2)) / po_norm
    return po_w

def calc_po_weight_c_align( csth, sigma=0.5 ):
    po_norm   = 2*sigma*erf(1/(2*sigma))
    po_w = (np.sqrt(2.0/np.pi))*np.exp(-(csth**2)/(2*sigma**2)) / po_norm
    return po_w



po_norm   = 2*sigma*erf(1/(2*sigma))
ncsamp = 1000 
csth_samples = np.arange(ncsamp)*2/ncsamp - 1.0
po_distribution = np.zeros(ncsamp)
for i in range(ncsamp):
    po_distribution[i] = calc_po_weight_c_align( csth_samples[i], sigma)

po_distribution *= 1.0/np.max(po_distribution)
plt.plot(csth_samples, po_distribution)
plt.xlabel( r'$\cos\theta$')
plt.ylabel( r'weight')
plt.xlim([-1,1])
plt.ylim([0,1.1])
plt.title( "Preferred orientation weight distribution")
plt.savefig(outpath + f'po_distribution_sigma{sigma}.jpg')

#debye-waller factor
rms = 0.33
B = 8*np.pi*np.pi*(rms**2)



# size of reciprocal lattice grid
nh, nk, nl = 7, 7, 7
Fhkl = np.array((nh,nk,nl))

reflections = []
for ih in range(-(nh//2),nh//2+1):
    for ik in range(-(nk//2),nk//2+1):
        for il in range(-(nl//2),nl//2+1):
            # ignore point at the origin
            if (ik==0)and(ih==0)and(il==0): continue

            # calculate q vector
            q = ih*astar + ik*bstar + il*cstar
            modq = np.sqrt(np.sum(q*q))

            #oxygen scattering factor
            sf = wk_sf(modq, wk)

            #structure factor
            Fhkl = 0.0
            for atom in xyz:
                Fhkl += sf*np.exp(2*np.pi*1j*np.dot(q,atom)) * np.exp(-B*modq*modq/4)

            #preferred orientation
            if sigma<10.0:
                csth = np.dot(q,cstar)/(modq*cstar_mod)
                po_weight = calc_po_weight_c_align(csth, sigma)
            else:
                po_weight = 1.0
    
            # add the reflection to the list
            reflections.append([ih,ik,il,modq,np.real(Fhkl), np.imag(Fhkl), po_weight*np.abs(Fhkl)**2])

reflections = np.array(reflections).T
print(reflections.shape)
reflections = reflections[:, reflections[3,:].argsort()]
reflections[6,:] *= 1.0/(reflections[3,:]**2)
reflections[6,:] *= 1.0/(reflections[6,0])
for i in range(8): print( "check", reflections[:3,i], reflections[3,i], reflections[-1,i])

plt.figure()


#turn the reflection list into a powder pattern
powder, be = np.histogram( reflections[3], bins = 2000, weights=reflections[6])

# normalized powder
npowder = powder/np.max(powder)
pp = [191,216,245,378,489]
pplabels = ["100", "002", "101","102","110"]
offx, offy = -0.0122, -0.1

print(np.min(powder),np.max(powder))
x = 0.5*(be[1:]+be[:-1]) 
plt.plot(x, npowder)
plt.ylabel( r'I(q)')
plt.xlabel( r'q (A$^{-1}$)')
plt.xlim([0.2,0.55])
plt.title( f'Hexagonal ice powder; Preferred orientation sigma = {sigma}')
for i, pl in enumerate(pplabels):
   plt.annotate( f'({pl})', (x[pp[i]]+offx, npowder[pp[i]]+offy), rotation=90)
plt.savefig(outpath + f'Hexagonal_ice_powder_sigma{sigma}.jpg'  )
plt.draw()
plt.show()


#write a list of the first few peak heights
ipeak = np.where(npowder>1e-13)
print("first few peaks")
for i in range(7): print(f'({ipeak[0][i]})'+"  q = ", x[ipeak[0][i]], " ; I = ", npowder[ipeak[0][i]])

#hex peak positions
pp = [191,216,245,378,489]
pplabels = ["100", "002", "101","102","110"]
### cubic pp labels "111", "220", "311", "331", "422" ### 
print("Ice peak ratios")
for i in range(5): print(f'({pp[i]})'+"  q = ", x[pp[i]], " ; I = ", npowder[pp[i]])



"""
# SOME OLD CODE THAT DOESN'T WORK

unique_refl = []
refl_intensities = []
mult = []
for i in range(reflections.shape[1]):
    r = reflections[3,i]
    if r not in unique_refl:
        unique_refl.append(r)
        refl_i = np.sum(reflections[6,reflections[3,:]==r])
        print( i, r, refl_i)
        refl_intensities.append(refl_i)

refl_intensities = np.array(refl_intensities)
unique_refl = np.array(unique_refl)

plt.plot( unique_refl, refl_intensities,".")
plt.xlim([0,3])    
plt.vlines( unique_refl, unique_refl*0.0, refl_intensities)


#offx, offy = 0.025, 0.0
#hkl_list = ['100', '110', '200']
#for i, hkl in enumerate(hkl_list):
#   plt.annotate( '('+hkl+')', (refl_intensities[i]+offx, refl_intensities[i]+offy))
"""

