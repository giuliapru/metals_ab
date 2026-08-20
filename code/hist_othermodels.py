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

neutral_frac_min = 0.1 #minimum neutral fraction in the gas particle
temp_lim = np.log10(2e4) #maximum temperature
density_lim = -2 #minimum density
metallicity_lim = -4 #minimum metallicity
#Z_solar = 0.0196
Z_solar = 0.0127
###############

run = 'KobayashiSNII' #KobayashiSNII
cond_hr = 0.5
hist = True
dla_gas_only = 'True' #if True, only plot DLA-like gas
gal = str(sys.argv[2])


figs, axs = mlib.plot_Sodini_all()

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
cond_neutral = (hi_density/h_density) > neutral_frac_min
DLA_cut_cond = cond_neutral & (np.log10(h_density)>density_lim) & (np.log10(temperature)<temp_lim) & (np.log10(metallicity/Z_solar)>metallicity_lim)

if dla_gas_only:
    print('Number of ptcs: %.2f' % len(temperature), ' checking condition cold is of the same dimension: %.2f' %len(DLA_cut_cond))
    print('In galaxy %s, the number of DLA-like gas particles is %.2f' % (gal, len(DLA_cut_cond[DLA_cut_cond==True])))
    h_density = h_density[DLA_cut_cond]
    hi_density = hi_density[DLA_cut_cond]
    carbon_density = carbon_density[DLA_cut_cond]
    oxygen_density = oxygen_density[DLA_cut_cond]
    silicon_density = silicon_density[DLA_cut_cond]
    iron_density = iron_density[DLA_cut_cond]
    temperature = temperature[DLA_cut_cond]
    volume = volume[DLA_cut_cond]
    gasmass = gasmass[DLA_cut_cond]
    metallicity = metallicity[DLA_cut_cond]
    print('In galaxy %s, the number of DLA-like gas particles after applying the condition is %d' % (gal, len(temperature)))


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


c_fe_bins = np.linspace(-1.5, 1.8, 300)
o_fe_bins = np.linspace(-1.5, 2.5, 300)
si_c_bins = np.linspace(-1.5, 1., 300)
si_fe_bins = np.linspace(-0.5, 1.5, 300)
si_o_bins = np.linspace(-1.5, 1.2, 300)
c_o_bins = np.linspace(-1.5, 1.2, 300)

mlib.hist_2d(np.array(c_fe_allg), np.array(o_fe_allg), axs[0], x_bins = c_fe_bins, y_bins=o_fe_bins)
mlib.hist_2d(np.array(c_fe_allg), np.array(si_c_allg), axs[1], x_bins = c_fe_bins, y_bins=si_c_bins)
mlib.hist_2d(np.array(o_fe_allg), np.array(si_fe_allg), axs[2], x_bins = o_fe_bins, y_bins=si_fe_bins)
mlib.hist_2d(np.array(c_o_allg), np.array(si_o_allg), axs[3], x_bins = c_o_bins, y_bins=si_o_bins)




###### phase diagram
'''figg, axx = plt.subplots(1, 1, figsize=(6,6))

mlib.hist_2d(np.log10(h_density), np.log10(temperature), axx, x_bins = 300, y_bins=300)
axx.set_xlabel(r'$\log_{10}(n_{\rm H})$ [cm$^{-3}$]')
axx.set_ylabel(r'$\log_{10}(T)$ [K]')
axx.set_xlim(-6,5)
axx.set_ylim(1,7.5)
'''
if run == 'IllustrisSNII':
    figs.savefig('/home/gpruto/metal_ab/images/all_gas/yields_comp/%s_all_z%.1f_hist_IllustrisSNII.png' % (gal, red), bbox_inches='tight')
    #figg.savefig('/home/gpruto/metal_ab/images/all_gas/yields_comp/%s_all_z%.1f_temp_vs_density_IllustrisSNII.png' % (gal, red), bbox_inches='tight')

if run == 'KobayashiSNII':
    figs.savefig('/home/gpruto/metal_ab/images/all_gas/yields_comp/%s_all_z%.1f_hist_KobayashiSNII.png' % (gal, red), bbox_inches='tight')
    figs.savefig('/home/gpruto/metal_ab/images/paper/%s_all_z%.1f_hist_KobayashiSNII.png' % (gal, red), bbox_inches='tight')

    #figg.savefig('/home/gpruto/metal_ab/images/all_gas/yields_comp/%s_all_z%.1f_temp_vs_density_KobayashiSNII.png' % (gal, red), bbox_inches='tight')
