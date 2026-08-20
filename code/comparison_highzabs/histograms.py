'''This code writes the histogram data of Si/O and C/O for the DLA-like gas particles. 
these are the data to use to compare to high-z (Nakane and co) observations. '''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('/home/gpruto/CGM_galaxies/paper.style')
import sys
sys.path.append('/home/gpruto/CGM_ref_analysis/code')
import lib
from haloes_class import TargetHalo
sys.path.append('/home/gpruto/metal_ab/code')
import metals_lib as mlib


#### SNAP
snap = int(sys.argv[1]) #92, 68, 51
if snap == 188:
    red = 3
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
coords, volume, redshift, gasmass, _, _, h_density, hi_density, carbon_density, oxygen_density, silicon_density, iron_density, temperature, metallicity = targethalo.gas_properties(snap, cond_hr, 1., all=True, metals=True, temperature=True, metallicity=True)
cond_neutral = (hi_density/h_density) > neutral_frac_min
DLA_cut_cond = cond_neutral & (np.log10(h_density)>density_lim) & (np.log10(temperature)<temp_lim) & (np.log10(metallicity/Z_solar)>metallicity_lim)


#### SELECTING DLA-LIKE GAS CELLS
si_o_allg = []
c_o_allg = []

print('Number of ptcs: %.2f' % len(temperature), ' checking condition cold is of the same dimension: %.2f' %len(DLA_cut_cond))
print('In galaxy %s, the number of possible DLA gas particles is %.2f' % (gal, len(DLA_cut_cond[DLA_cut_cond==True])))
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
print('In galaxy %s, the number of cold gas particles after applying the condition is %d' % (gal, len(temperature)))

si_o = np.log10(silicon_density/oxygen_density) - mlib.Si_O_solar
c_o = np.log10(carbon_density/oxygen_density) - mlib.C_O_solar

si_o_allg.extend(np.array(si_o).ravel().tolist())
c_o_allg.extend(np.array(c_o).ravel().tolist())
###################



##### METAL ABUNDANCES PLOTS
#fake plot to get the histogram data
fig, ax = plt.subplots(figsize=(6,5))
si_o_bins = np.linspace(-1.5, 2, 400)
c_o_bins = np.linspace(-1.3, 2, 300)
hist4, hist4_xedges, hist4_yedges = mlib.hist_2d(np.array(c_o_allg), np.array(si_o_allg), ax, x_bins = c_o_bins, y_bins=si_o_bins, output=True)


### SAVE HISTOGRAM DATA TO FILE
output_file = '/home/gpruto/metal_ab/code/2dhistograms/z=%d/%s/comphighz_x_HI<%.1f_n_H>%d_T<%.1f_met>%.1f.txt' % (red, gal, neutral_frac_min, density_lim, temp_lim, metallicity_lim)

with open(output_file, 'w') as f:
    f.write('#x_bin y_bin hist4\n')
    for i in range(len(hist4)):
        for j in range(len(hist4[i])):
            f.write('%d %d %d\n' %(i, j, hist4[i][j]))

