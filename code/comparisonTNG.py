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

#### SNII YIELDS
infile_Arepo = '/cephfs/gpruto/CGM_ref/arepo/data/Arepo_GFM_Tables/Yields/SNII.hdf5'
infile_TNG = '/home/gpruto/TNG_Yields/SNII.hdf5'
infiles = [infile_Arepo, infile_TNG]


#Species_names = [b'Hydrogen' b'Helium' b'Carbon' b'Nitrogen' b'Oxygen' b'Neon'
# b'Magnesium' b'Silicon' b'Sulphur' b'Calcium' b'Iron']

for i in range(len(infiles)):
    infile = infiles[i]
    with h5py.File(infile,'r') as f:
        n_masses = int(f['Number_of_masses'][()])    # scalar -> python int
        print('Number_of_masses =', n_masses)
        if i==0:
            Arepo_masses_or = f['Masses'][:]
            print('Masses =', Arepo_masses_or)
        else:
            TNG_masses_or = f['Masses'][:]
            print('Masses =', TNG_masses_or)
        
        n_metallicities = int(f['Number_of_metallicities'][()])
        if i==0:
            Arepo_metallicities = f['Metallicities'][:]
            print('Metallicities =', Arepo_metallicities)
        else:
            TNG_metallicities = f['Metallicities'][:]
            print('Metallicities =', TNG_metallicities)
        
        elements = f['Species_names'][:]
        print('Species_names =', elements)

#consider only metallicities present in both files
Arepo_metallicities = np.array([z for z in Arepo_metallicities if z in TNG_metallicities])
TNG_metallicities = np.array([z for z in TNG_metallicities if z in Arepo_metallicities])
print('Metallicities present in both files =', Arepo_metallicities, TNG_metallicities)

Arepo_masses = np.array([m for m in Arepo_masses_or if m in TNG_masses_or])
TNG_masses = np.array([m for m in TNG_masses_or if m in Arepo_masses_or])
idx_Arepo_mass = [int(np.where(Arepo_masses_or == m)[0][0]) for m in Arepo_masses]
idx_TNG_mass = [int(np.where(TNG_masses_or == m)[0][0]) for m in TNG_masses]

print('Masses present in both files =', Arepo_masses, TNG_masses)
print('Indices of masses in Arepo file =', idx_Arepo_mass)
print('Indices of masses in TNG file =', idx_TNG_mass)


Arepo_yields = np.zeros((len(Arepo_metallicities), len(elements), len(Arepo_masses)))
TNG_yields = np.zeros((len(TNG_metallicities), len(elements), len(TNG_masses)))

for m in range(len(Arepo_masses)):
    for z in range(len(Arepo_metallicities)):
        with h5py.File(infile_Arepo,'r') as f:
            if z==0:
                Arepo_yields[z,:, m] = f['Yields'][f'Z_%.3f' % Arepo_metallicities[z]]['Yield'][:, idx_Arepo_mass[m]]
            else:
                Arepo_yields[z,:, m] = f['Yields'][f'Z_%.2f' % Arepo_metallicities[z]]['Yield'][:, idx_Arepo_mass[m]]
        with h5py.File(infile_TNG,'r') as f:
            if z==0:
                TNG_yields[z,:, m] = f['Yields'][f'Z_%.3f' % TNG_metallicities[z]]['Yield'][:, idx_TNG_mass[m]]
            else:
                TNG_yields[z, :, m] = f['Yields'][f'Z_%.2f' % TNG_metallicities[z]]['Yield'][:, idx_TNG_mass[m]]
            

Arepo_C = Arepo_yields[:, 2]
Arepo_O = Arepo_yields[:,4]
Arepo_Si = Arepo_yields[:,7]
Arepo_Fe = Arepo_yields[:, 10]

TNG_C = TNG_yields[:, 2]
TNG_O = TNG_yields[:, 4]
TNG_Si = TNG_yields[:, 7]
TNG_Fe = TNG_yields[:, 10]


fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax = ax.flatten()

c = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
for z in range(len(Arepo_metallicities)):
    for m in range(len(Arepo_masses)):
        ax[0].scatter(Arepo_C[z, m], TNG_C[z, m], c=c[z], s=Arepo_masses[m])
        ax[1].scatter(Arepo_O[z, m], TNG_O[z, m], c=c[z], s=Arepo_masses[m])
        ax[2].scatter(Arepo_Si[z, m], TNG_Si[z, m], c=c[z], s=Arepo_masses[m])
        ax[3].scatter(Arepo_Fe[z, m], TNG_Fe[z, m], c=c[z], s=Arepo_masses[m])

        if z==0:
            ax[3].scatter([], [], c='black', s=Arepo_masses[m], label=r'Mass = %d M$_\odot$' % Arepo_masses[m])
    
    ax[2].scatter([], [], c=c[z], s=50, label=r'Z = %.3f' % Arepo_metallicities[z])

ax[2].legend()
ax[3].legend()

for i in range(len(ax)):
    ax[i].plot([-0.5, 10], [-0.5, 10], c='black', ls='--')
    ax[i].set_xlabel('Thesan-Zoom yields')
    ax[i].set_ylabel('TNG yields')

ax[0].set_xlim(-0.5, 7)
ax[0].set_ylim(-0.5, 7)
ax[1].set_xlim(-0.5, 9)
ax[1].set_ylim(-0.5, 9)
ax[2].set_xlim(-0.5, 1)
ax[2].set_ylim(-0.5, 1)
ax[3].set_xlim(-0.5, 1)
ax[3].set_ylim(-0.5, 1)

ax[0].set_title('Carbon yields')
ax[1].set_title('Oxygen yields')
ax[2].set_title('Silicon yields')
ax[3].set_title('Iron yields')

fig.savefig('/home/gpruto/metal_ab/images/yield_comp_TNG.png', bbox_inches='tight', dpi=300)


figs, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()

axs[0].set_xlim(-1.5, 3)
axs[0].set_ylim(-1.5, 3.5)
axs[1].set_xlim(-2, 2.5)
axs[1].set_ylim(-2.7, 2)
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


gal = 'g578'
snap=51
run='fiducial'
cond_hr = 0.5
targethalo = TargetHalo(gal, run)
targethalo.read_haloes(snap, 0)
halo_mass = targethalo.data[snap]['mass200']*1e10/lib.h
coords, volume, redshift, gasmass, _, _, h_density, hi_density, carbon_density, oxygen_density, silicon_density, iron_density, temperature = targethalo.gas_properties(snap, cond_hr, 1., all=True, metals=True, temperature=True, metallicity  = False)

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


for z in range(len(Arepo_metallicities)):
    for m in range(len(Arepo_masses)):
        axs[0].scatter(np.log10(Arepo_C[z,m]/Arepo_Fe[z,m]) - mlib.C_Fe_solar, np.log10(Arepo_O[z,m]/Arepo_Fe[z,m]) - mlib.O_Fe_solar, edgecolors=c[z], fc='white', s=Arepo_masses[m])
        axs[0].scatter(np.log10(TNG_C[z,m]/TNG_Fe[z,m]) - mlib.C_Fe_solar, np.log10(TNG_O[z,m]/TNG_Fe[z,m]) - mlib.O_Fe_solar, edgecolors=c[z], fc='white', s=TNG_masses[m], marker='s')

        axs[1].scatter(np.log10(Arepo_C[z,m]/Arepo_Fe[z,m]) - mlib.C_Fe_solar, np.log10(Arepo_Si[z,m]/Arepo_C[z,m]) - mlib.Si_C_solar, edgecolors=c[z], fc='white', s=Arepo_masses[m])
        axs[1].scatter(np.log10(TNG_C[z,m]/TNG_Fe[z,m]) - mlib.C_Fe_solar, np.log10(TNG_Si[z,m]/TNG_C[z,m]) - mlib.Si_C_solar, edgecolors=c[z], fc='white', s=TNG_masses[m], marker='s')

        axs[2].scatter(np.log10(Arepo_O[z,m]/Arepo_Fe[z,m]) - mlib.O_Fe_solar, np.log10(Arepo_Si[z,m]/Arepo_Fe[z,m]) - mlib.Si_Fe_solar, edgecolors=c[z], fc='white', s=Arepo_masses[m])
        axs[2].scatter(np.log10(TNG_O[z,m]/TNG_Fe[z,m]) - mlib.O_Fe_solar, np.log10(TNG_Si[z,m]/TNG_Fe[z,m]) - mlib.Si_Fe_solar, edgecolors=c[z], fc='white', s=TNG_masses[m], marker='s')

        axs[3].scatter(np.log10(Arepo_C[z,m]/Arepo_O[z,m]) - mlib.C_O_solar, np.log10(Arepo_Si[z,m]/Arepo_O[z,m]) - mlib.Si_O_solar, edgecolors=c[z], fc='white', s=Arepo_masses[m])
        axs[3].scatter(np.log10(TNG_C[z,m]/TNG_O[z,m]) - mlib.C_O_solar, np.log10(TNG_Si[z,m]/TNG_O[z,m]) - mlib.Si_O_solar, edgecolors=c[z], fc='white', s=TNG_masses[m], marker='s')

        if z ==0:
            axs[3].scatter([], [], c='black', s=Arepo_masses[m], label=r'%d M$_\odot$' % Arepo_masses[m])
            if m==0:
                axs[1].scatter([], [], fc='white', edgecolors='black', s=50, label=r'Arepo', marker='o')
                axs[1].scatter([], [], fc='white', edgecolors='black', s=50, label=r'TNG', marker='s')

    axs[2].scatter([], [], c=c[z], s=50, label=r'Z = %.3f' % Arepo_metallicities[z])

axs[0].set_xlabel(r'[C/Fe]')
axs[0].set_ylabel(r'[O/Fe]')
axs[1].set_xlabel(r'[C/Fe]')
axs[1].set_ylabel(r'[Si/C]')
axs[2].set_xlabel(r'[O/Fe]')
axs[2].set_ylabel(r'[Si/Fe]')
axs[3].set_xlabel(r'[C/O]')
axs[3].set_ylabel(r'[Si/O]')

axs[3].legend(loc='lower right')
axs[2].legend()
axs[1].legend()
figs.savefig('/home/gpruto/metal_ab/images/yield_ratios_comp_TNG.png', bbox_inches='tight', dpi=300)

'''print('Number_of_metallicities =', n_metallicities)
n_species = int(f['Number_of_species'][()])
print('Number_of_species =', n_species)
species_names = f['Species_names'][:]
print('Species_names =', species_names)
yield_names = f['Yield_names'][:]
print('Yield_names =', yield_names)
print('Just to check... ', f['Yields']['Z_0.02']['Yield'])

'''