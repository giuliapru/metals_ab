import numpy as np
import matplotlib.pyplot as plt
plt.style.use('/home/gpruto/CGM_galaxies/paper.style')
import h5py
import sys
import os
sys.path.append('/home/gpruto/CGM_ref_analysis/code')
import lib
from haloes_class import TargetHalo
sys.path.append('/home/gpruto/metal_ab/code')
import metals_lib as mlib
from tqdm.notebook import tqdm as progressbar
from mpl_toolkits.mplot3d import Axes3D
from scipy import spatial
from shapely.geometry import Point, Polygon
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
import cmcrameri.cm as cmc
import matplotlib.colors


figs, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()

norm = matplotlib.colors.Normalize(vmin=0, vmax=5)
colours=[]

for i in range(7):
    colours.append(cmc.lipari(norm(i)))

axs[0].set_xlim(-1.5, 3)
axs[0].set_ylim(-1.5, 3.5)
axs[1].set_xlim(-2, 2.5)
axs[1].set_ylim(-2.7, 1.3)
axs[2].set_xlim(-2, 3.8)
axs[2].set_ylim(-1, 1.8)
axs[3].set_xlim(-1.4, 1.2)
axs[3].set_ylim(-3.5, 1.3)

si_o_allg = []
c_o_allg = []
c_fe_allg = []
o_fe_allg = []
si_fe_allg = []
si_c_allg = []
c_fe_allg = []


gal = 'g519761'
snap=51
run='fiducial'
cond_hr = 0.5
targethalo = TargetHalo(gal, run)
targethalo.read_haloes(snap, 0)
halo_mass = targethalo.data[snap]['mass200']*1e10/lib.h
coords, volume, redshift, gasmass, _, _, h_density, hi_density, carbon_density, oxygen_density, silicon_density, iron_density, temperature = targethalo.gas_properties(snap, cond_hr, 1., all=True, metals=True, temperature=True)

neutral_oxygen_density = oxygen_density*hi_density/h_density

si_o = np.log10(silicon_density/oxygen_density) - mlib.Si_O_solar
c_o = np.log10(carbon_density/oxygen_density) - mlib.C_O_solar
c_fe = np.log10(carbon_density/iron_density) - mlib.C_Fe_solar
o_fe = np.log10(oxygen_density/iron_density) - mlib.O_Fe_solar
si_fe = np.log10(silicon_density/iron_density) - mlib.Si_Fe_solar
si_c = np.log10(silicon_density/carbon_density) - mlib.Si_C_solar
print('The length of the silicon/oxygen abundance array is %.2f' % len(si_o))

si_o_allg.extend(np.array(si_o).ravel().tolist())
c_o_allg.extend(np.array(c_o).ravel().tolist())
c_fe_allg.extend(np.array(c_fe).ravel().tolist())
o_fe_allg.extend(np.array(o_fe).ravel().tolist())
si_fe_allg.extend(np.array(si_fe).ravel().tolist())
si_c_allg.extend(np.array(si_c).ravel().tolist())

mlib.hist_2d(np.array(c_fe_allg), np.array(o_fe_allg), axs[0], x_bins = 300, y_bins=300)
mlib.hist_2d(np.array(c_fe_allg), np.array(si_c_allg), axs[1], x_bins = 300, y_bins=300)
mlib.hist_2d(np.array(o_fe_allg), np.array(si_fe_allg), axs[2], x_bins = 300, y_bins=300)
mlib.hist_2d(np.array(c_o_allg), np.array(si_o_allg), axs[3], x_bins = 300, y_bins=300)

for aa in [axs]:
    aa[0].set_xlabel(r'[C/Fe]')
    aa[0].set_ylabel(r'[O/Fe]')
    aa[1].set_xlabel(r'[C/Fe]')
    aa[1].set_ylabel(r'[Si/C]')
    aa[2].set_xlabel(r'[O/Fe]')
    aa[2].set_ylabel(r'[Si/Fe]')
    aa[3].set_xlabel(r'[C/O]')
    aa[3].set_ylabel(r'[Si/O]')

#####SN II yields from Portinari et al. 1998
carbon_Portinari = np.zeros((len(mlib.Portinari_metallicities), len(mlib.Portinari_masses)))
oxygen_Portinari = np.zeros((len(mlib.Portinari_metallicities), len(mlib.Portinari_masses)))
silicon_Portinari = np.zeros((len(mlib.Portinari_metallicities), len(mlib.Portinari_masses)))
iron_Portinari = np.zeros((len(mlib.Portinari_metallicities), len(mlib.Portinari_masses)))

#check we're considering the right elements
#print(mlib.Portinari_elements[3:5])
#print(mlib.Portinari_elements[7:10])
#print(mlib.Portinari_elements[-4])
#print(mlib.Portinari_elements[-1])

for met in range(len(mlib.Portinari_metallicities)):
    for mass in range(len(mlib.Portinari_masses)):
        carbon_Portinari[met, mass] = np.sum(mlib.Portinari_yields[met, mass, 3:5])
        oxygen_Portinari[met, mass] = np.sum(mlib.Portinari_yields[met, mass, 7:10])
        silicon_Portinari[met, mass] = mlib.Portinari_yields[met, mass, -4]
        iron_Portinari[met, mass] = mlib.Portinari_yields[met, mass, -1]

sizes = [9, 12, 15, 20, 30, 40, 60, 100, 120]
for i in range(len(mlib.Portinari_metallicities)):
    axs[0].scatter(np.log10(carbon_Portinari[i,:]/iron_Portinari[i,:]) - mlib.C_Fe_solar, np.log10(oxygen_Portinari[i,:]/iron_Portinari[i,:]) - mlib.O_Fe_solar, c=colours[i],s=sizes, label='Z = %.4f' %(mlib.Portinari_metallicities[i]))
    axs[1].scatter(np.log10(carbon_Portinari[i,:]/iron_Portinari[i,:]) - mlib.C_Fe_solar, np.log10(silicon_Portinari[i,:]/carbon_Portinari[i,:]) - mlib.Si_C_solar, c=colours[i],s=sizes)
    axs[2].scatter(np.log10(oxygen_Portinari[i,:]/iron_Portinari[i,:]) - mlib.O_Fe_solar, np.log10(silicon_Portinari[i,:]/iron_Portinari[i,:]) - mlib.Si_Fe_solar, c=colours[i],s=sizes)
    axs[3].scatter(np.log10(carbon_Portinari[i,:]/oxygen_Portinari[i,:]) - mlib.C_O_solar, np.log10(silicon_Portinari[i,:]/oxygen_Portinari[i,:]) - mlib.Si_O_solar, c=colours[i],s=sizes)

for j in range(len(mlib.Portinari_masses)):
    axs[1].scatter([],[], c='k', s=sizes[j], label='M = %d M$_\odot$' %(mlib.Portinari_masses[j]))

axs[0].legend()
axs[1].legend(loc='lower left')
figs.savefig('/home/gpruto/metal_ab/images/SNII_tracks_yields.png', dpi=300, bbox_inches='tight')


#### SNIa yields from Thielemann et al. 2003
fig_snia, ax_snia = plt.subplots(2, 2, figsize=(10, 10))
ax_snia = ax_snia.flatten()
carbon_snia = np.sum(mlib.Thie_2003_yields[0:2])
oxygen_snia = np.sum(mlib.Thie_2003_yields[4:7])
silicon_snia = np.sum(mlib.Thie_2003_yields[16:19])
iron_snia = np.sum(mlib.Thie_2003_yields[50:54])

##check we're considering the right elements
#print(mlib.Thie_2003_elements[0:2])
#print(mlib.Thie_2003_yields[0:2])
#print(mlib.Thie_2003_elements[4:7])
#print(mlib.Thie_2003_yields[4:7])
#print(mlib.Thie_2003_elements[16:19])
#print(mlib.Thie_2003_yields[16:19])
#print(mlib.Thie_2003_elements[50:54])
#print(mlib.Thie_2003_yields[50:54])
mlib.hist_2d(np.array(c_fe_allg), np.array(o_fe_allg), ax_snia[0], x_bins = 300, y_bins=300)
mlib.hist_2d(np.array(c_fe_allg), np.array(si_c_allg), ax_snia[1], x_bins = 300, y_bins=300)
mlib.hist_2d(np.array(o_fe_allg), np.array(si_fe_allg), ax_snia[2], x_bins = 300, y_bins=300)
mlib.hist_2d(np.array(c_o_allg), np.array(si_o_allg), ax_snia[3], x_bins = 300, y_bins=300)


for aa in [axs, ax_snia]:
    aa[0].set_xlabel(r'[C/Fe]')
    aa[0].set_ylabel(r'[O/Fe]')
    aa[1].set_xlabel(r'[C/Fe]')
    aa[1].set_ylabel(r'[Si/C]')
    aa[2].set_xlabel(r'[O/Fe]')
    aa[2].set_ylabel(r'[Si/Fe]')
    aa[3].set_xlabel(r'[C/O]')
    aa[3].set_ylabel(r'[Si/O]')

    aa[0].scatter(np.log10(carbon_snia/iron_snia) - mlib.C_Fe_solar, np.log10(oxygen_snia/iron_snia) - mlib.O_Fe_solar, c='tab:red', s=50, marker='x')
    aa[1].scatter(np.log10(carbon_snia/iron_snia) - mlib.C_Fe_solar, np.log10(silicon_snia/carbon_snia) - mlib.Si_C_solar, c='tab:red', s=50, marker='x')
    aa[2].scatter(np.log10(oxygen_snia/iron_snia) - mlib.O_Fe_solar, np.log10(silicon_snia/iron_snia) - mlib.Si_Fe_solar, c='tab:red', s=50, marker='x', label='SNIa')
    aa[3].scatter(np.log10(carbon_snia/oxygen_snia) - mlib.C_O_solar, np.log10(silicon_snia/oxygen_snia) - mlib.Si_O_solar, c='tab:red', s=50, marker='x')

    aa[0].set_xlim(-2.5, 3)
    aa[0].set_ylim(-2.3, 3.5)
    aa[1].set_xlim(-2.5, 2.5)
    aa[1].set_ylim(-2.7, 1.8)
    aa[2].set_xlim(-2.3, 3.8)
    aa[2].set_ylim(-1, 2)
    aa[3].set_xlim(-1.4, 1.2)
    aa[3].set_ylim(-3.5, 1.5)

    aa[2].legend()

fig_snia.savefig('/home/gpruto/metal_ab/images/SNIa_tracks_yields.png', dpi=300, bbox_inches='tight')
figs.savefig('/home/gpruto/metal_ab/images/SNII_and_SNIa_tracks_yields.png', dpi=300, bbox_inches='tight')