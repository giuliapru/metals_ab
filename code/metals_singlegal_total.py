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

run = 'fiducial' 
cond_hr = 0.5
hist = True
cold_gas_only = str(sys.argv[2]) == 'True' #if True, only plot gas with T < 10^4 K, otherwise plot all gas
gal = str(sys.argv[3])
metal_cut = str(sys.argv[4]) == 'True' #if True, only plot gas with metallicity < 0.1 Z_solar, otherwise plot all gas

if metal_cut:
    metal_up = 0.1 * Z_solar #upper metallicity

factor_mod_si = 1




figs, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()

for aa in [axs]:
    mlib.plot_Sodini(mlib.CFe_Sodini, mlib.OFe_Sodini, mlib.CFe_err_Sodini, mlib.OFe_err_Sodini, mlib.CFe_limit_type, mlib.OFe_limit_type, aa[0], color='red', special=True, label='Sodini+24')
    mlib.plot_Sodini(mlib.CFe_Becker, mlib.OFe_Becker, mlib.CFe_Becker_err, mlib.OFe_Becker_err, mlib.CFe_Becker_limit_type, mlib.OFe_Becker_limit_type, aa[0], color='blue', label='Becker+20')
    mlib.plot_Sodini(mlib.CFe_Poudel_2018, mlib.OFe_Poudel_2018, mlib.CFe_Poudel_2018_err, mlib.OFe_Poudel_2018_err, mlib.CFe_Poudel_2018_limit_type, mlib.OFe_Poudel_2018_limit_type, aa[0], color='green', label='Poudel+18,+20')
    mlib.plot_Sodini(mlib.CFe_Poudel[:2], mlib.OFe_Poudel[:2], mlib.CFe_Poudel_err[:2], mlib.OFe_Poudel_err[:2], mlib.CFe_Poudel_limit_type[:2], mlib.OFe_Poudel_limit_type[:2], aa[0], color='green')
    aa[0].set_xlabel(r'[C/Fe]')
    aa[0].set_ylabel(r'[O/Fe]')

    mlib.plot_Sodini(mlib.CFe_Sodini, mlib.SiC_Sodini, mlib.CFe_err_Sodini, mlib.SiC_err_Sodini, mlib.CFe_limit_type, mlib.SiC_limit_type, aa[1], color = 'red', special=True)
    mlib.plot_Sodini(mlib.CFe_Becker, mlib.SiC_Becker, mlib.CFe_Becker_err, mlib.SiC_Becker_err, mlib.CFe_Becker_limit_type, mlib.SiC_Becker_limit_type, aa[1], color='blue')
    mlib.plot_Sodini(mlib.CFe_Poudel_2018, mlib.SiC_Poudel_2018, mlib.CFe_Poudel_2018_err, mlib.SiC_Poudel_2018_err, mlib.CFe_Poudel_2018_limit_type, mlib.SiC_Poudel_2018_limit_type, aa[1], color='green')
    mlib.plot_Sodini(mlib.CFe_Poudel[:2], mlib.SiC_Poudel[:2], mlib.CFe_Poudel_err[:2], mlib.SiC_Poudel_err[:2], mlib.CFe_Poudel_limit_type[:2], mlib.SiC_Poudel_limit_type[:2], aa[1], color='green')
    aa[1].set_xlabel(r'[C/Fe]')
    aa[1].set_ylabel(r'[Si/C]')

    mlib.plot_Sodini(mlib.OFe_Sodini, mlib.SiFe_Sodini, mlib.OFe_err_Sodini, mlib.SiFe_err_Sodini, mlib.OFe_limit_type, mlib.SiFe_limit_type, aa[2], color='red', special=True)
    mlib.plot_Sodini(mlib.OFe_Becker, mlib.SiFe_Becker, mlib.OFe_Becker_err, mlib.SiFe_Becker_err, mlib.OFe_Becker_limit_type, mlib.SiFe_Becker_limit_type, aa[2], color='blue')
    mlib.plot_Sodini(mlib.OFe_Poudel_2018, mlib.SiFe_Poudel_2018, mlib.OFe_Poudel_2018_err, mlib.SiFe_Poudel_2018_err, mlib.OFe_Poudel_2018_limit_type, mlib.SiFe_Poudel_2018_limit_type, aa[2], color='green')
    mlib.plot_Sodini(mlib.OFe_Poudel[:2], mlib.SiFe_Poudel[:2], mlib.OFe_Poudel_err[:2], mlib.SiFe_Poudel_err[:2], mlib.OFe_Poudel_limit_type[:2], mlib.SiFe_Poudel_limit_type[:2], aa[2], color='green')
    aa[2].set_xlabel(r'[O/Fe]')
    aa[2].set_ylabel(r'[Si/Fe]')

    mlib.plot_Sodini(mlib.CO_Sodini, mlib.SiO_Sodini, mlib.CO_err_Sodini, mlib.SiO_err_Sodini, mlib.CO_limit_type, mlib.SiO_limit_type, aa[3], color='red', special=True)
    mlib.plot_Sodini(mlib.CO_Becker, mlib.SiO_Becker, mlib.CO_Becker_err, mlib.SiO_Becker_err, mlib.CO_Becker_limit_type, mlib.SiO_Becker_limit_type, aa[3], color='blue')
    mlib.plot_Sodini(mlib.CO_Poudel_2018, mlib.SiO_Poudel_2018, mlib.CO_Poudel_2018_err, mlib.SiO_Poudel_2018_err, mlib.CO_Poudel_2018_limit_type, mlib.SiO_Poudel_2018_limit_type, aa[3], color='green')
    mlib.plot_Sodini(mlib.CO_Poudel[:2], mlib.SiO_Poudel[:2], mlib.CO_Poudel_err[:2], mlib.SiO_Poudel_err[:2], mlib.CO_Poudel_limit_type[:2], mlib.SiO_Poudel_limit_type[:2], aa[3], color='green')
    aa[3].set_xlabel(r'[C/O]')
    aa[3].set_ylabel(r'[Si/O]')




si_o_allg = []
c_o_allg = []
c_fe_allg = []
o_fe_allg = []
si_fe_allg = []
si_c_allg = []
c_fe_allg = []


targethalo = TargetHalo(gal, run)
targethalo.read_haloes(snap, 0)
halo_mass = targethalo.data[snap]['mass200']*1e10/lib.h
coords, volume, redshift, gasmass, _, _, h_density, hi_density, carbon_density, oxygen_density, silicon_density, iron_density, temperature, metallicity = targethalo.gas_properties(snap, cond_hr, 1., all=True, metals=True, temperature=True, metallicity=True)

if cold_gas_only:
    condition_cold = temperature < 1e4
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
    condition_metal = metallicity < metal_up
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

mlib.plot_models(mlib.CFe0_WW, mlib.OFe0_WW, axs[0], color='lightblue')
mlib.plot_models(mlib.CFe1_WW, mlib.SiC1_WW, axs[1], color='lightblue', label= 'PopII - WW95')
mlib.plot_models(mlib.OFe2_WW, mlib.SiFe2_WW, axs[2], color='lightblue')
mlib.plot_models(mlib.CO3_WW, mlib.SiO3_WW, axs[3], color='lightblue')


for aa in [axs]:
    aa[0].set_xlim(-1.3, 1.8)
    aa[0].set_ylim(-1.5, 2)
    aa[1].set_xlim(-1.3, 2)
    aa[1].set_ylim(-1., 1)
    aa[2].set_xlim(-1, 2.3)
    aa[2].set_ylim(-0.7, 1.5)
    aa[3].set_xlim(-1.4, 1)
    aa[3].set_ylim(-1.25, 1.35)


#put the legend in a separate box below the 4 plots
axs[0].legend()
axs[1].legend()
figs.subplots_adjust(bottom=0.2)
figs.subplots_adjust(wspace=0.3, hspace=0.3)

if cold_gas_only and not metal_cut:
    if factor_mod_si != 1:
        figs.savefig('/home/gpruto/metal_ab/images/all_gas/cold_gas_only/%s_all_z%.1f_hist_coldgasonly_modsi.png' % (gal, red), bbox_inches='tight')
        plt.close()
    else:
        figs.savefig('/home/gpruto/metal_ab/images/all_gas/cold_gas_only/%s_all_z%.1f_hist_coldgasonly.png' % (gal, red), bbox_inches='tight')
        plt.close()

elif metal_cut and not cold_gas_only:
    if factor_mod_si != 1:
        figs.savefig('/home/gpruto/metal_ab/images/all_gas/low_met_only/%s_all_z%.1f_hist_metalcut_modsi.png' % (gal, red), bbox_inches='tight')
        plt.close()
    else:
        figs.savefig('/home/gpruto/metal_ab/images/all_gas/low_met_only/%s_all_z%.1f_hist_metalcut.png' % (gal, red), bbox_inches='tight')
        plt.close()

elif metal_cut and cold_gas_only:
    if factor_mod_si != 1:
        figs.savefig('/home/gpruto/metal_ab/images/all_gas/cold_low_met/%s_all_z%.1f_hist_cold_metalcut_modsi.png' % (gal, red), bbox_inches='tight')
        plt.close()
    else:
        figs.savefig('/home/gpruto/metal_ab/images/all_gas/cold_low_met/%s_all_z%.1f_hist_cold_metalcut.png' % (gal, red), bbox_inches='tight')
        plt.close()

else:
    if factor_mod_si != 1:
        figs.savefig('/home/gpruto/metal_ab/images/all_gas/all_ptcs/%s_all_z%.1f_hist_modsi.png' % (gal, red), bbox_inches='tight')
        plt.close()
    else:
        figs.savefig('/home/gpruto/metal_ab/images/all_gas/all_ptcs/%s_all_z%.1f_hist.png' % (gal, red), bbox_inches='tight')
        plt.close()
