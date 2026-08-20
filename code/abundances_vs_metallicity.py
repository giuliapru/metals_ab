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


#### SNAP
snap = int(sys.argv[1]) #92, 68, 51
if snap == 92:
    red = 5
elif snap == 68:
    red = 6
elif snap == 51:
    red = 7
elif snap == 39:
    red = 8

Z_solar = 0.0196

#### PARAMETERS
run =  ['fiducial'] #['fiducial', 'IllustrisSNII', 'KobayashiSNII']
colors = ['black', 'tab:orange', 'tab:blue']
cond_hr = 0.5
hist = True
relative_ab = False
#cold_gas_only = str(sys.argv[2]) == 'True' #if True, only plot gas with T < 10^4 K, otherwise plot all gas
dla_cut = True
gal = str(sys.argv[2])
#metal_cut = str(sys.argv[4]) == 'True' #if True, only plot gas with metallicity < 0.1 Z_solar, otherwise plot all gas

'''if metal_cut:
    metal_up = -1.5
    metal_down = -4'''

#factor_mod_si = 1


fe_h_bins = np.linspace(-6, 0, 20)
fe_h_binsc = 0.5*(fe_h_bins[1:]+fe_h_bins[:-1])

if relative_ab:
    figs, axs = plt.subplots(6, 4, figsize=(22, 30))
    axs[0, 0].set_ylabel(r'[C/Fe]')
    axs[1, 0].set_ylabel(r'[O/Fe]')
    axs[2, 0].set_ylabel(r'[Si/C]')
    axs[3, 0].set_ylabel(r'[Si/Fe]')
    axs[4, 0].set_ylabel(r'[C/O]')
    axs[5, 0].set_ylabel(r'[Si/O]')
    for a in range(len(axs[0])):    
        axs[-1, a].set_xlabel(r'[Fe/H]')

else:
    figs, axs = plt.subplots(3, 1, figsize=(6, 15)) #plt.subplots(3, 4, figsize=(22, 15))
    axs = np.reshape(axs, (3, 1))
    axs[0, 0].set_ylabel(r'[C/H]')
    axs[1, 0].set_ylabel(r'[O/H]')
    axs[2, 0].set_ylabel(r'[Si/H]')
    for a in range(len(axs[0])):
        axs[-1, a].set_xlabel(r'[Fe/H]')



if relative_ab:
    si_o_perc = np.zeros((3, len(fe_h_bins)-1, 3))
    c_o_perc = np.zeros((3, len(fe_h_bins)-1, 3))
    c_fe_perc = np.zeros((3, len(fe_h_bins)-1, 3))
    o_fe_perc = np.zeros((3, len(fe_h_bins)-1, 3))
    si_fe_perc = np.zeros((3, len(fe_h_bins)-1, 3))
    si_c_perc = np.zeros((3, len(fe_h_bins)-1, 3))
    c_fe_perc = np.zeros((3, len(fe_h_bins)-1, 3))
    fe_h_perc = np.zeros((3, len(fe_h_bins)-1, 3))
else:
    si_h_perc = np.zeros((3, len(fe_h_bins)-1, 3))
    c_h_perc = np.zeros((3, len(fe_h_bins)-1, 3))
    o_h_perc = np.zeros((3, len(fe_h_bins)-1, 3))
    fe_h_perc = np.zeros((3, len(fe_h_bins)-1, 3))

for r in range(len(run)):
    
    targethalo = TargetHalo(gal, run[r])
    targethalo.read_haloes(snap, 0)
    halo_mass = targethalo.data[snap]['mass200']*1e10/lib.h
    coords, volume, redshift, gasmass, _, _, h_density, hi_density, carbon_density, oxygen_density, silicon_density, iron_density, temperature, metallicity = targethalo.gas_properties(snap, cond_hr, 1., all=True, metals=True, temperature=True, metallicity=True)
    neutral_fraction = hi_density/h_density

    if dla_cut:
        condition_dla = (neutral_fraction > 0.1) & (h_density > 1e-2) & (temperature < 1e4)
        h_density = h_density[condition_dla]
        hi_density = hi_density[condition_dla]
        carbon_density = carbon_density[condition_dla]
        oxygen_density = oxygen_density[condition_dla]
        silicon_density = silicon_density[condition_dla]
        iron_density = iron_density[condition_dla]
        temperature = temperature[condition_dla]
        volume = volume[condition_dla]
        gasmass = gasmass[condition_dla]
        metallicity = metallicity[condition_dla]

    '''if cold_gas_only:
    condition_cold = temperature < 5e3
    print('Number of ptcs: %.2f' % len(temperature), ' checking condition cold is of the same dimension: %.2f' %len(condition_cold))
    print('In galaxy %s, the number of cold gas particles is %.2f' % (gal, len(condition_cold[condition_cold==True])))
    h_density = h_density[condition_cold]
    hi_density = hi_density[condition_cold]
    carbon_density = carbon_density[condition_cold]
    oxygen_density = oxygen_density[condition_cold]
    silicon_density = silicon_density[condition_cold]
    iron_density = iron_density[condition_cold]
    temperature = temperature[condition_cold]
    volume = volume[condition_cold]
    gasmass = gasmass[condition_cold]
    metallicity = metallicity[condition_cold]
    print('In galaxy %s, the number of cold gas particles after applying the condition is %d' % (gal, len(temperature)))

    if metal_cut:
    print('Number of ptcs: %.2f' % len(temperature))
    iron_ab = np.log10(iron_density/h_density) - mlib.Fe_sun
    condition_metal = (iron_ab < metal_up) & (iron_ab > metal_down)
    print('In galaxy %s, the number of gas particles with metallicity < 0.1 Z_solar is %.2f' % (gal, len(condition_metal[condition_metal==True])))
    h_density = h_density[condition_metal]
    hi_density = hi_density[condition_metal]
    carbon_density = carbon_density[condition_metal]
    oxygen_density = oxygen_density[condition_metal]
    silicon_density = silicon_density[condition_metal]
    iron_density = iron_density[condition_metal]
    temperature = temperature[condition_metal]
    volume = volume[condition_metal]
    gasmass = gasmass[condition_metal]
    metallicity = metallicity[condition_metal]
    print('In galaxy %s, the number of gas particles after applying the metallicity cut is %d' % (gal, len(temperature)))
    '''
    neutral_oxygen_density = oxygen_density*hi_density/h_density

    if relative_ab: #relative abundances
        si_o_allg = []
        c_o_allg = []
        c_fe_allg = []
        o_fe_allg = []
        si_fe_allg = []
        si_c_allg = []
        c_fe_allg = []
        fe_h_allg = []

        si_o = np.log10(silicon_density/oxygen_density) - mlib.Si_O_solar
        c_o = np.log10(carbon_density/oxygen_density) - mlib.C_O_solar
        c_fe = np.log10(carbon_density/iron_density) - mlib.C_Fe_solar
        o_fe = np.log10(oxygen_density/iron_density) - mlib.O_Fe_solar
        si_fe = np.log10(silicon_density/iron_density) - mlib.Si_Fe_solar
        si_c = np.log10(silicon_density/carbon_density) - mlib.Si_C_solar
        fe_h = np.log10(iron_density/h_density) - mlib.Fe_sun
        print('The length of the silicon/oxygen abundance array is %.2f' % len(si_o))


        si_o_allg.extend(np.array(si_o).ravel().tolist())
        c_o_allg.extend(np.array(c_o).ravel().tolist())
        c_fe_allg.extend(np.array(c_fe).ravel().tolist())
        o_fe_allg.extend(np.array(o_fe).ravel().tolist())
        si_fe_allg.extend(np.array(si_fe).ravel().tolist())
        si_c_allg.extend(np.array(si_c).ravel().tolist())
        fe_h_allg.extend(np.array(fe_h).ravel().tolist())

        condition_fe_h = np.array(fe_h_allg) > -6
        mlib.hist_2d(np.array(fe_h_allg)[condition_fe_h],np.array(c_fe_allg)[condition_fe_h], axs[0,r], x_bins = 300, y_bins=300)
        mlib.hist_2d(np.array(fe_h_allg)[condition_fe_h], np.array(o_fe_allg)[condition_fe_h], axs[1,r], x_bins = 300, y_bins=300)
        mlib.hist_2d(np.array(fe_h_allg)[condition_fe_h],np.array(si_c_allg)[condition_fe_h], axs[2,r], x_bins = 300, y_bins=300)
        mlib.hist_2d(np.array(fe_h_allg)[condition_fe_h], np.array(si_fe_allg)[condition_fe_h], axs[3,r], x_bins = 300, y_bins=300)
        mlib.hist_2d(np.array(fe_h_allg)[condition_fe_h], np.array(c_o_allg)[condition_fe_h], axs[4,r], x_bins = 300, y_bins=300)
        mlib.hist_2d(np.array(fe_h_allg)[condition_fe_h], np.array(si_o_allg)[condition_fe_h], axs[5,r], x_bins = 300, y_bins=300)
        
        for b in range(len(fe_h_bins)-1):
            condition_bin = (np.array(fe_h_allg) > fe_h_bins[b]) & (np.array(fe_h_allg) < fe_h_bins[b+1])
            si_o_perc[r, b] = np.percentile(np.array(si_o_allg)[condition_bin], [16,50,84])
            c_o_perc[r,b] = np.percentile(np.array(c_o_allg)[condition_bin], [16,50,84])
            c_fe_perc[r,b] = np.percentile(np.array(c_fe_allg)[condition_bin], [16,50,84])
            o_fe_perc[r,b] = np.percentile(np.array(o_fe_allg)[condition_bin], [16,50,84])
            si_fe_perc[r,b] = np.percentile(np.array(si_fe_allg)[condition_bin], [16,50,84])
            si_c_perc[r,b] = np.percentile(np.array(si_c_allg)[condition_bin], [16,50,84])
            c_fe_perc[r,b] = np.percentile(np.array(c_fe_allg)[condition_bin], [16,50,84])

        axs[0,3].plot(fe_h_binsc, c_fe_perc[r,:,1], color=colors[r], label=run[r])
        axs[1,3].plot(fe_h_binsc, o_fe_perc[r,:,1], color=colors[r])
        axs[2,3].plot(fe_h_binsc, si_c_perc[r,:,1], color=colors[r])
        axs[3,3].plot(fe_h_binsc, si_fe_perc[r,:,1], color=colors[r])
        axs[4,3].plot(fe_h_binsc, c_o_perc[r,:,1], color=colors[r])
        axs[5,3].plot(fe_h_binsc, si_o_perc[r,:,1], color=colors[r])

        axs[0,3].fill_between(fe_h_binsc, c_fe_perc[r,:,0], c_fe_perc[r,:,2], color=colors[r], alpha=0.3)
        axs[1,3].fill_between(fe_h_binsc, o_fe_perc[r,:,0], o_fe_perc[r,:,2], color=colors[r], alpha=0.3)
        axs[2,3].fill_between(fe_h_binsc, si_c_perc[r,:,0], si_c_perc[r,:,2], color=colors[r], alpha=0.3)
        axs[3,3].fill_between(fe_h_binsc, si_fe_perc[r,:,0], si_fe_perc[r,:,2], color=colors[r], alpha=0.3)
        axs[4,3].fill_between(fe_h_binsc, c_o_perc[r,:,0], c_o_perc[r,:,2], color=colors[r], alpha=0.3)
        axs[5,3].fill_between(fe_h_binsc, si_o_perc[r,:,0], si_o_perc[r,:,2], color=colors[r], alpha=0.3)

    else: #single elements
        si_h_allg = []
        c_h_allg = []
        o_h_allg = []
        fe_h_allg = []

        si_h = np.log10(silicon_density/h_density) - mlib.Si_sun
        c_h = np.log10(carbon_density/h_density) - mlib.C_sun
        o_h = np.log10(oxygen_density/h_density) - mlib.O_sun
        fe_h = np.log10(iron_density/h_density) - mlib.Fe_sun
        
        si_h_allg.extend(np.array(si_h).ravel().tolist())
        c_h_allg.extend(np.array(c_h).ravel().tolist())
        o_h_allg.extend(np.array(o_h).ravel().tolist())
        fe_h_allg.extend(np.array(fe_h).ravel().tolist())

        condition_fe_h = np.array(fe_h_allg) > -6
        mlib.hist_2d(np.array(fe_h_allg)[condition_fe_h],np.array(c_h_allg)[condition_fe_h], axs[0,r], x_bins = 300, y_bins=300)
        mlib.hist_2d(np.array(fe_h_allg)[condition_fe_h], np.array(o_h_allg)[condition_fe_h], axs[1,r], x_bins = 300, y_bins=300)
        mlib.hist_2d(np.array(fe_h_allg)[condition_fe_h],np.array(si_h_allg)[condition_fe_h], axs[2,r], x_bins = 300, y_bins=300)

        '''for b in range(len(fe_h_bins)-1):
            condition_bin = (np.array(fe_h_allg) > fe_h_bins[b]) & (np.array(fe_h_allg) < fe_h_bins[b+1])
            si_h_perc[r, b] = np.percentile(np.array(si_h_allg)[condition_bin], [16,50,84])
            c_h_perc[r,b] = np.percentile(np.array(c_h_allg)[condition_bin], [16,50,84])
            o_h_perc[r,b] = np.percentile(np.array(o_h_allg)[condition_bin], [16,50,84])
            fe_h_perc[r,b] = np.percentile(np.array(fe_h_allg)[condition_bin], [16,50,84])

        axs[0,3].plot(fe_h_binsc, c_h_perc[r,:,1], color=colors[r], label=run[r])
        axs[1,3].plot(fe_h_binsc, o_h_perc[r,:,1], color=colors[r])
        axs[2,3].plot(fe_h_binsc, si_h_perc[r,:,1], color=colors[r])

        axs[0,3].fill_between(fe_h_binsc, c_h_perc[r,:,0], c_h_perc[r,:,2], color=colors[r], alpha=0.3)
        axs[1,3].fill_between(fe_h_binsc, o_h_perc[r,:,0], o_h_perc[r,:,2], color=colors[r], alpha=0.3)
        axs[2,3].fill_between(fe_h_binsc, si_h_perc[r,:,0], si_h_perc[r,:,2], color=colors[r], alpha=0.3)
'''

#axs[0,3].legend()

'''axs[0,0].set_title('Thesan-Zoom')
axs[0,1].set_title('Illustris')
axs[0,2].set_title('Kobayashi')
axs[0,3].set_title('Comparison')
'''

if relative_ab:
    for r in range(len(axs[0])):
        axs[0,r].set_ylim(-1.5, 3.5)
        axs[1,r].set_ylim(-2.5, 3.5)
        axs[2,r].set_ylim(-2.7, 1.1)
        axs[3,r].set_ylim(-1,2)
        axs[4,r].set_ylim(-1.7, 1)
        axs[5,r].set_ylim(-3, 2)
        for a in range(len(axs)):
            axs[a,r].set_xlim(-6, 0)

    figs.savefig('/home/gpruto/metal_ab/images/all_gas/yields_comp/abun_vs_met_%s_z%d_histandlines.png' %(gal, red), bbox_inches='tight',dpi=300)

else:
    for r in range(len(axs[0])):
        axs[0,r].set_ylim(-6, 0)
        axs[1,r].set_ylim(-6, 0)
        axs[2,r].set_ylim(-6, 0)
        for a in range(len(axs)):
            axs[a,r].set_xlim(-6, 0)

    figs.savefig('/home/gpruto/metal_ab/images/all_gas/yields_comp/abun_vs_met_%s_z%d_singleel.png' %(gal, red), bbox_inches='tight', dpi=300)