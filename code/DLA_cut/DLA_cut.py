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
from scipy.spatial import cKDTree

#### SNAP
snap = int(sys.argv[1]) #92, 68, 51
if snap == 188:
    red = 3
if snap == 129:
    red = 4
if snap == 92:
    red = 5
elif snap == 68:
    red = 6
elif snap == 51:
    red = 7
elif snap == 39:
    red = 8
elif snap == 30:
    red = 9


#### PARAMETERS
run = 'fiducial'
cond_hr = 0.5
hist = True
gal = str(sys.argv[2])

neutral_frac_min = 0.1 #minimum neutral fraction in the gas particle
temp_lim = np.log10(2e4) #maximum temperature
density_lim = float(sys.argv[3]) #minimum density
metallicity_lim = -4 #minimum metallicity
#Z_solar = 0.0196
Z_solar = 0.0127
###############


#### READ DATA
targethalo = TargetHalo(gal, run)
targethalo.read_haloes(snap, 0)
halo_mass = targethalo.data[snap]['mass200']*1e10/lib.h
coords, volume, redshift, gasmass, _, _, h_density, hi_density, carbon_density, oxygen_density, silicon_density, iron_density, temperature, metallicity, dust_to_gas = targethalo.gas_properties(snap, cond_hr, 1., all=True, metals=True, temperature=True, metallicity=True, dust_to_gas = True)
cond_neutral = (hi_density/h_density) > neutral_frac_min
DLA_cut_cond = cond_neutral & (np.log10(h_density)>density_lim) & (np.log10(temperature)<temp_lim) & (np.log10(metallicity/Z_solar)>metallicity_lim)


'''
##### Plotting histograms of temperature, density, metallicity for the selected region
fig, ax = plt.subplots(1,4, figsize=(18,4.5))

x_hi_bins = np.linspace(0,1, 300)
n_h_bins = np.linspace(-6, 5, 300)
T_bins = np.linspace(1, 7.5, 300)
met_bins = np.linspace(-15, 1, 300)

hist1_tot, hist1_xedges, hist1_yedges = mlib.hist_2d(np.log10(h_density), (hi_density/h_density), ax[0], x_bins = n_h_bins, y_bins=x_hi_bins, alpha=0.3, output=True)
hist2_tot, hist2_xedges, hist2_yedges = mlib.hist_2d(np.log10(temperature), (hi_density/h_density), ax[1], x_bins = T_bins, y_bins=x_hi_bins, alpha=0.3, output=True)
hist3_tot, hist3_xedges, hist3_yedges = mlib.hist_2d(np.log10(metallicity/Z_solar), (hi_density/h_density), ax[2], x_bins = met_bins, y_bins=x_hi_bins, alpha=0.3, output=True)
hist4_tot, hist4_xedges, hist4_yedges = mlib.hist_2d(np.log10(h_density), np.log10(temperature), ax[3], x_bins = n_h_bins, y_bins=T_bins, alpha=0.3, output=True)

hist1, hist1_xedges, hist1_yedges = mlib.hist_2d(np.log10(h_density[DLA_cut_cond]), (hi_density[DLA_cut_cond]/h_density[DLA_cut_cond]), ax[0], x_bins = n_h_bins, y_bins=x_hi_bins, output=True)
hist2, hist2_xedges, hist2_yedges = mlib.hist_2d(np.log10(temperature[DLA_cut_cond]), (hi_density[DLA_cut_cond]/h_density[DLA_cut_cond]), ax[1], x_bins = T_bins, y_bins=x_hi_bins, output=True)
hist3, hist3_xedges, hist3_yedges = mlib.hist_2d(np.log10(metallicity[DLA_cut_cond]/Z_solar), (hi_density[DLA_cut_cond]/h_density[DLA_cut_cond]), ax[2], x_bins = met_bins, y_bins=x_hi_bins, output=True)
hist4, hist4_xedges, hist4_yedges = mlib.hist_2d(np.log10(h_density[DLA_cut_cond]), np.log10(temperature[DLA_cut_cond]), ax[3], x_bins = n_h_bins, y_bins=T_bins, output=True)

ax[0].plot([-6, 5], [0.1, 0.1], color='k', ls='--', lw=2)
ax[0].vlines(density_lim, 0, 1, color='k', ls='--', lw=2)

ax[1].plot([1, 7.5], [0.1, 0.1], color='k', ls='--', lw=2)
ax[1].vlines(temp_lim, 0, 1, color='k', ls='--', lw=2)

ax[2].plot([-15, 1], [0.1, 0.1], color='k', ls='--', lw=2)
ax[2].vlines(metallicity_lim, 0, 1, color='k', ls='--', lw=2)

ax[3].plot([-6, 5], [temp_lim, temp_lim], color='k', ls='--', lw=2)
ax[3].plot([density_lim, density_lim], [1, 7.5], color='k', ls='--', lw=2)

ax[0].set_xlabel(r'$\log_{10} (n_{\rm H}$ [cm$^{-3}$])')
ax[1].set_xlabel(r'$\log_{10} (T$ [K])')
ax[2].set_xlabel(r'$\log_{10} (Z/Z_{\odot}$)')
ax[3].set_xlabel(r'$\log_{10} (n_{\rm H}$ [cm$^{-3}$])')

ax[0].set_ylabel(r'$n_{\rm HI}/n_{\rm H}$')
ax[1].set_ylabel(r'$n_{\rm HI}/n_{\rm H}$')
ax[2].set_ylabel(r'$n_{\rm HI}/n_{\rm H}$')
ax[3].set_ylabel(r'$\log_{10} (T$ [K])')

ax[0].set_ylim(-0.01,1.01)
ax[1].set_ylim(-0.01,1.01)
ax[2].set_ylim(-0.01,1.01)
ax[3].set_ylim(1, 7.5)

ax[0].set_xlim(-6,5)
ax[1].set_xlim(1,7.5)
ax[2].set_xlim(-15,1)
ax[3].set_xlim(-6,5)

fig.savefig('/home/gpruto/metal_ab/images/DLA_cut/x_HI<%.1f_n_H>%d_T<%.1f_met>%.1f/dla_cuts_%s_z%d_1x4.png' %(neutral_frac_min, density_lim, temp_lim, metallicity_lim, gal, redshift), bbox_inches='tight', dpi=300)


#### SAVE HISTOGRAM DATA TO FILE
output_file = '/home/gpruto/metal_ab/code/2dhistograms/z=%d/%s/dlacuts_hist_x_HI<%.1f_n_H>%d_T<%.1f_met>%.1f.txt' % (red, gal, neutral_frac_min, density_lim, temp_lim, metallicity_lim)

with open(output_file, 'w') as f:
    f.write('#x_bin y_bin hist1_tot hist2_tot hist3_tot hist4_tot hist1 hist2 hist3 hist4\n') #_tot for the histograms with all gas ptcs, histn for the dla-like
    for i in range(len(hist1)):
        for j in range(len(hist1[i])):
            f.write('%d %d %d %d %d %d %d %d %d %d\n' %(i, j, hist1_tot[i][j], hist2_tot[i][j], hist3_tot[i][j], hist4_tot[i][j], hist1[i][j], hist2[i][j], hist3[i][j], hist4[i][j]))
###############################
'''



#### SELECTING DLA-LIKE GAS CELLS
si_o_allg = []
c_o_allg = []
c_fe_allg = []
o_fe_allg = []
si_fe_allg = []
si_c_allg = []
c_fe_allg = []

print('Number of ptcs: %.2f' % len(temperature), ' checking condition cold is of the same dimension: %.2f' %len(DLA_cut_cond))
print('In galaxy %s, the number of possible DLA gas particles is %.2f' % (gal, len(DLA_cut_cond[DLA_cut_cond==True])))
h_density = h_density[DLA_cut_cond]
hi_density = hi_density[DLA_cut_cond]
carbon_density = carbon_density[DLA_cut_cond]
oxygen_density = oxygen_density[DLA_cut_cond]
silicon_density = silicon_density[DLA_cut_cond]
iron_density = iron_density[DLA_cut_cond]
temperature = temperature[DLA_cut_cond]
gasmass = gasmass[DLA_cut_cond]
metallicity = metallicity[DLA_cut_cond]
dla_volume = volume[DLA_cut_cond]
dla_coords = coords[DLA_cut_cond]
dust_to_gas = dust_to_gas[DLA_cut_cond]
print('In galaxy %s, the number of cold gas particles after applying the condition is %d' % (gal, len(temperature)))

si_o = np.log10(silicon_density/oxygen_density) - mlib.Si_O_solar
c_o = np.log10(carbon_density/oxygen_density) - mlib.C_O_solar
c_fe = np.log10(carbon_density/iron_density) - mlib.C_Fe_solar
o_fe = np.log10(oxygen_density/iron_density) - mlib.O_Fe_solar
si_fe = np.log10(silicon_density/iron_density) - mlib.Si_Fe_solar
si_c = np.log10(silicon_density/carbon_density) - mlib.Si_C_solar

si_o_allg.extend(np.array(si_o).ravel().tolist())
c_o_allg.extend(np.array(c_o).ravel().tolist())
c_fe_allg.extend(np.array(c_fe).ravel().tolist())
o_fe_allg.extend(np.array(o_fe).ravel().tolist())
si_fe_allg.extend(np.array(si_fe).ravel().tolist())
si_c_allg.extend(np.array(si_c).ravel().tolist())


#writing proerties of the DLA-like gas cells to a text file
outfile = '/home/gpruto/metal_ab/code/DLA_cut/z=%d/%s/dla_cuts_x_HI>%.1f_n_H>%d_T<%.1f_met>%.1f.txt' %(red, gal, neutral_frac_min, density_lim, temp_lim, metallicity_lim)
with open(outfile, 'w') as f:
    f.write('#x y z mass volume n_H n_HI [C/O] [Si/O] [C/Fe] [O/Fe] [Si/Fe] [Si/C] dust_to_gas\n')
    for i in range(len(coords[DLA_cut_cond])):
        f.write('%.4f %.4f %.4f %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e\n' %(dla_coords[i][0], dla_coords[i][1], dla_coords[i][2], gasmass[i], dla_volume[i], h_density[i], hi_density[i], c_o[i], si_o[i], c_fe[i], o_fe[i], si_fe[i], si_c[i], dust_to_gas[i]))
########################


'''
##### METAL ABUNDANCES PLOTS
figs, axs = mlib.plot_Sodini_all() #plot Sodini data in the 4 plots

#plotting the 2d histograms
c_fe_bins = np.linspace(-1.5, 1.8, 300)
o_fe_bins = np.linspace(-1.5, 2.5, 300)
si_c_bins = np.linspace(-1.5, 1., 300)
si_fe_bins = np.linspace(-0.5, 1.2, 300)
si_o_bins = np.linspace(-1.5, 1.2, 300)
c_o_bins = np.linspace(-1.2, 1.2, 300)
hist1, hist1_xedges, hist1_yedges = mlib.hist_2d(np.array(c_fe_allg), np.array(o_fe_allg), axs[0], x_bins = c_fe_bins, y_bins=o_fe_bins, output=True)
hist2, hist2_xedges, hist2_yedges = mlib.hist_2d(np.array(c_fe_allg), np.array(si_c_allg), axs[1], x_bins = c_fe_bins, y_bins=si_c_bins, output=True)
hist3, hist3_xedges, hist3_yedges = mlib.hist_2d(np.array(o_fe_allg), np.array(si_fe_allg), axs[2], x_bins = o_fe_bins, y_bins=si_fe_bins, output=True)
hist4, hist4_xedges, hist4_yedges = mlib.hist_2d(np.array(c_o_allg), np.array(si_o_allg), axs[3], x_bins = c_o_bins, y_bins=si_o_bins, output=True)


for aa in [axs]:
    aa[0].set_xlim(-1.3, 1.8)
    aa[0].set_ylim(-1.5, 2)
    aa[1].set_xlim(-1.3, 2)
    aa[1].set_ylim(-1., 1)
    aa[2].set_xlim(-1, 2.3)
    aa[2].set_ylim(-0.7, 1.5)
    aa[3].set_xlim(-1.4, 1)
    aa[3].set_ylim(-1.25, 1.35)

axs[0].legend()
axs[1].legend()
figs.subplots_adjust(bottom=0.2)
figs.subplots_adjust(wspace=0.3, hspace=0.3)

figs.savefig('/home/gpruto/metal_ab/images/all_gas/neutral_f>%.1f/T<%.0e_n>%.1f/met>-4/%s_z%d_hist.png' %(neutral_frac_min, 10**temp_lim, density_lim, gal, red), bbox_inches='tight', dpi=300)


### SAVE METAL ABUNDANCE HISTOGRAM DATA TO FILE
output_file = '/home/gpruto/metal_ab/code/2dhistograms/z=%d/%s/Sodini_hist_x_HI<%.1f_n_H>%d_T<%.1f_met>%.1f.txt' % (red, gal, neutral_frac_min, density_lim, temp_lim, metallicity_lim)

with open(output_file, 'w') as f:
    f.write('#x_bin y_bin hist1 hist2 hist3 hist4\n')
    for i in range(len(hist1)):
        for j in range(len(hist1[i])):
            f.write('%d %d %d %d %d %d\n' %(i, j, hist1[i][j], hist2[i][j], hist3[i][j], hist4[i][j]))
###########################




### PROPERTIES OF DLA-LIKE GAS CELLS

### read data about haloes
halocoord, mass200, massDM, halomass, haloradius, redshift = lib.read_halos(gal, run, snap)
centre = halocoord[0]


dla_idx = np.flatnonzero(DLA_cut_cond)
#dla_coords = coords[DLA_cut_cond]
#dla_volume = volume[DLA_cut_cond] #already done above
dla_vol_frac = np.sum(dla_volume)/np.sum(volume)
print(len(dla_coords), len(dla_volume))
dla_tree = cKDTree(dla_coords)

in_rvir = np.zeros(len(dla_coords), dtype=bool)
in_1p5rvir = np.zeros(len(dla_coords), dtype=bool)

for halo_center, rvir in zip(halocoord, haloradius):
    candidate_idx = dla_tree.query_ball_point(halo_center, r=1.5 * rvir)
    if not candidate_idx:
        continue

    candidate_idx = np.asarray(candidate_idx, dtype=int)
    delta = dla_coords[candidate_idx] - halo_center
    d2 = np.einsum('ij,ij->i', delta, delta)

    in_1p5rvir[candidate_idx] |= d2 < (1.5 * rvir)**2
    in_rvir[candidate_idx] |= d2 < rvir**2

in_rvir_sum = in_rvir.sum()
in_1p5rvir_sum = in_1p5rvir.sum()
    
print('The fraction of DLAs that are in haloes is: ', in_rvir_sum/np.sum(DLA_cut_cond))
print('The fraction of DLAs that are in 1.5xRvir of haloes is: ', in_1p5rvir_sum/np.sum(DLA_cut_cond))

output_file = '/home/gpruto/metal_ab/code/DLA_cut/dla_cuts_x_HI>%.1f_n_H>%d_T<%.1f_met>%.1f.txt' %(neutral_frac_min, density_lim, temp_lim, metallicity_lim)
with open(output_file, 'a') as f:
    if os.stat(output_file).st_size == 0:
        f.write('#galaxy redshift fraction_DLA fraction_in_Rvir fraction_in_1.5Rvir\n')
    #if the given galaxy at the given redshift is present, don't write
    if not any((line.startswith('%s %d' %(gal, redshift))) for line in open(output_file)):
        f.write('%s %d %.4f %.4f %.4f\n' %(gal, redshift, np.sum(DLA_cut_cond)/len(DLA_cut_cond), in_rvir_sum/np.sum(DLA_cut_cond), in_1p5rvir_sum/np.sum(DLA_cut_cond)))


print('The volume fraction of DLAs that are in haloes is: ', np.sum(dla_volume[in_rvir])/np.sum(dla_volume))
print('The volume fraction of DLAs that are in 1.5xRvir of haloes is: ', np.sum(dla_volume[in_1p5rvir])/np.sum(dla_volume))

output_file = '/home/gpruto/metal_ab/code/DLA_cut/dla_cuts_volume_x_HI>%.1f_n_H>%d_T<%.1f_met>%.1f.txt' %(neutral_frac_min, density_lim, temp_lim, metallicity_lim)
with open(output_file, 'a') as f:
    if os.stat(output_file).st_size == 0:
        f.write('#galaxy redshift fraction_DLA fraction_in_Rvir fraction_in_1.5Rvir\n')
    #if the given galaxy at the given redshift is present, don't write
    if not any((line.startswith('%s %d' %(gal, redshift))) for line in open(output_file)):
        f.write('%s %d %.4e %.4e %.4e\n' %(gal, redshift, dla_vol_frac, np.sum(dla_volume[in_rvir])/np.sum(dla_volume), np.sum(dla_volume[in_1p5rvir])/np.sum(dla_volume)))'''