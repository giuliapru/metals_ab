'''This is to check that the simulations using the new yields are producing somewhat expected results.'''


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('/home/gpruto/CGM_galaxies/paper.style')
import h5py
import sys
import os
sys.path.append('/home/gpruto/CGM_ref_analysis/code')
import lib
from tqdm.notebook import tqdm as progressbar
from haloes_class import TargetHalo
from mpl_toolkits.mplot3d import Axes3D
from scipy import spatial
from shapely.geometry import Point, Polygon
from scipy.stats import gaussian_kde

snap = 51 #92, 68, 51
cond_hr = 0.5

C_sun = 8.47 - 12
O_sun = 8.73 - 12
Si_sun = 7.55 - 12
Fe_sun = 7.54 - 12

Si_O_solar = Si_sun - O_sun
Si_Fe_solar = Si_sun - Fe_sun
Si_C_solar = Si_sun - C_sun
C_O_solar = C_sun - O_sun
C_Fe_solar = C_sun - Fe_sun
O_Fe_solar = O_sun - Fe_sun


def add_contours(ax, x, y, color, levels=(0.001, 0.5), all=False):
    
    # Remove NaNs if needed
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    # KDE
    kde = gaussian_kde([x, y])

    # Grid to evaluate density
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kde(positions), xx.shape)

    # Normalize density (so levels behave nicely)
    f /= f.max()
    if all==True:
        levels = (0.00001, 0.5)

    ax.contour(xx, yy, f, levels=levels, colors=color, linewidths=3)



#### SODINI DATA ####
# --- [C/Fe] ---
CFe_Sodini = np.array([
 0.18,0.21,0.70,0.46,0.03,0.28,0.25,0.35,-0.05,0.08,0.24,0.11,
 -0.21,0.33,0.18,0.34, 0.60,0.04,0.39,0.33,0.26,0.54,0.23,0.22,
 0.50,0.21,0.41,0.38
])

CFe_err_Sodini = np.array([
 0.07,0.04,0.18,0.06,np.nan,np.nan,0.05,0.09,np.nan,np.nan,0.04,np.nan,
 0.19,0.19,0.04,np.nan,np.nan,0.03,0.05,0.07,0.03,0.16,0.08,0.07,
 np.nan,0.14,np.nan,np.nan
])

CFe_lowlim = np.array([True if np.isnan(CFe_err_Sodini[i]) else False for i in range(len(CFe_err_Sodini))])
CFe_limit_type = np.array([np.nan if not np.isnan(CFe_err_Sodini[i]) else 'lower' for i in range(len(CFe_err_Sodini))])

# --- [O/Fe] ---
OFe_Sodini = np.array([
 0.33,0.26,0.49,-0.07,0.03,0.74,0.36,0.44,0.21,0.76,0.27,0.49,
 0.29,0.35,0.40,0.25,-0.06,0.32,0.00,0.85,0.55,np.nan,0.18,0.22,
 0.38,0.41,0.56, 0.84
])

OFe_err_Sodini = np.array([
 0.06,0.04,0.10,0.06,np.nan,np.nan,0.05,0.08,0.22,np.nan,0.08,np.nan,
 0.16,0.11,0.03,np.nan,0.04,0.04,0.06,0.14,0.03,np.nan,0.05,0.05,
 np.nan,0.06,np.nan,np.nan
])

OFe_lowlim = np.array([True if np.isnan(OFe_err_Sodini[i]) else False for i in range(len(OFe_err_Sodini))])
OFe_limit_type = np.array([np.nan if not np.isnan(OFe_err_Sodini[i]) else 'lower' for i in range(len(OFe_err_Sodini))])


# --- [Si/Fe] ---
SiFe_Sodini = np.array([
 0.56,0.33,0.78,0.59,0.28,np.nan,0.37,0.46,0.11,0.39,0.50,0.45,
 0.04,0.67,0.44,0.56,0.65,0.43,0.51,0.35,0.34,0.52,0.54,0.58,
 0.58,0.50,0.95, 0.57
])

SiFe_err_Sodini = np.array([
 0.09,0.03,np.nan,0.06,0.10,np.nan,0.05,0.08,0.20,0.10,0.05,np.nan,
 0.16,0.08,0.04,0.03,0.04,0.04,0.09,0.04,0.03,0.08,0.05,0.04,
 np.nan,0.08,np.nan,np.nan
])

limit = np.array([True if np.isnan(SiFe_err_Sodini[i]) else False for i in range(len(SiFe_err_Sodini))])
SiFe_limit_type = np.array([np.nan if not np.isnan(SiFe_err_Sodini[i]) else 'lower' for i in range(len(SiFe_err_Sodini))])
SiFe_limit_type[np.where(SiFe_limit_type=='lower')[0][0]] = 'upper'


# --- [C/O] ---
CO_Sodini = np.array([
 -0.15,-0.05,0.21,0.53,np.nan,-0.46,-0.11,-0.09,-0.26,np.nan,-0.03,
 -0.38,-0.50,-0.02,-0.21,np.nan,0.66,-0.28,0.39,-0.52,-0.29,
 np.nan,0.05,0.00,0.12,-0.20,-0.15,-0.46
])

CO_err_Sodini = np.array([
 0.04,0.04,0.15,0.05,np.nan,0.07,0.03,0.04,0.11,np.nan,0.08,
 0.13,0.13,0.20,0.03,np.nan,np.nan,0.04,0.04,0.15,0.03,
 np.nan,0.07,0.08,0.21,0.13,0.10,0.27
])

CO_limit_type = np.array([np.nan if not np.isnan(CO_err_Sodini[i]) else 'lower' for i in range(len(CO_err_Sodini))])

# --- [Si/O] ---
SiO_Sodini = np.array([
 0.23,0.08,0.29,0.67,0.25,-0.24,0.01,0.02,-0.10,-0.37,0.23,
 -0.04,-0.25,0.32,0.04,0.31,0.71,0.11,0.51,-0.50,-0.21,
 np.nan,0.36,0.36,0.20,0.09,0.39,-0.27
])

SiO_err_Sodini = np.array([
 0.07,0.03,np.nan,0.05,np.nan,np.nan,0.02,0.03,0.09,np.nan,0.08,
 0.07,0.07,0.09,0.04,np.nan,0.04,0.04,0.08,0.14,0.03,
 np.nan,0.04,0.05,0.17,0.06,0.13,0.10
])

SiO_limit_type = np.array([np.nan if not np.isnan(SiO_err_Sodini[i]) else 'upper' for i in range(len(SiO_err_Sodini))])

# --- [Fe/O] ---
FeO_Sodini = np.array([
 -0.33,-0.26,-0.49,0.07,-0.03,-0.74,-0.36,-0.44,-0.21,-0.76,
 -0.27,-0.49,-0.29,-0.35,-0.40,-0.25,0.06,-0.32,0.00,-0.85,
 -0.55,np.nan,-0.18,-0.22,-0.38,-0.41,-0.56, -0.84
])

FeO_err_Sodini = np.array([
 0.06,0.04,0.10,0.06,np.nan,np.nan,0.05,0.08,0.22,np.nan,
 0.08,np.nan,0.16,0.11,0.03,np.nan,0.04,0.04,0.06,0.14,
 0.03,np.nan,0.05,0.05,np.nan,0.06,np.nan,np.nan
])
FeO_limit_type = np.array([np.nan if not np.isnan(FeO_err_Sodini[i]) else 'upper' for i in range(len(FeO_err_Sodini))])

# --- log N_CII ---
logN_CII = np.array([
 13.68,14.47,14.27,14.49,15.46,13.12,14.14,13.77,13.45, 14.33,
 14.31,13.18,13.17,14.22,14.53,15.24,14.91,14.09,13.57,13.91,14.11,
 14.64,14.31,14.52,13.03,14.00,13.47,13.84
])

logN_CII_err = np.array([
 0.03, 0.03,0.15,0.03,np.nan,0.06,0.01,0.03,0.07, np.nan,
 0.03,0.11,0.11,0.18,0.02,np.nan, np.nan, 0.02,0.02,0.07,0.02,
 0.16,0.06,0.07,0.12,0.13,0.09,0.25
])

N_CII_limit_type = np.array([np.nan if not np.isnan(logN_CII_err[i]) else 'lower' for i in range(len(logN_CII_err))])


# --- log N_SiII ---
logN_SiII = np.array([
 13.14,13.67,13.43,13.71,14.79,12.42,13.33,12.96,12.69,
 13.72,13.65,12.60,12.50,13.64,13.87,14.54,14.04,13.56,12.77,13.01,
 13.27,13.69,13.70,13.96,12.19,13.37,13.09,13.11
])

logN_SiII_err = np.array([
 0.07,0.02,np.nan,0.04,0.09,np.nan,0.01,0.02,0.04,
 0.07,0.04,0.03, 0.04,0.03,0.03,0.02,0.03, 0.03,0.07,0.03,
 0.02,0.07,0.04,0.03,0.04,0.06,0.12,0.05
])

SiIILimit_type = np.array([np.nan if not np.isnan(logN_SiII_err[i]) else 'upper' for i in range(len(logN_SiII_err))])


# --- Si/C ----
SiC_Sodini = logN_SiII - logN_CII - (Si_sun - C_sun)
SiC_err_Sodini = np.sqrt(logN_SiII_err**2 + logN_CII_err**2)
SiC_limit_type = np.array(['nan' if not np.isnan(logN_SiII_err[i]) and not np.isnan(logN_CII_err[i]) else 'upper' for i in range(len(logN_SiII_err))])

############


##### PLOTTING SODININI DATA #####
def plot_Sodini(x_data, y_data, x_err, y_err, x_limit, y_limit, ax, cc='red'):
    all_constrained = np.array([True if (x_limit[i]=='nan' and y_limit[i]=='nan') else False for i in range(len(x_limit))])
    ax.errorbar(x_data[all_constrained], y_data[all_constrained], xerr=x_err[all_constrained], yerr=y_err[all_constrained], fmt='o', color=cc, ecolor=cc, elinewidth=1, capsize=1)

    y_limited_low = np.array([True if (x_limit[i]=='nan' and y_limit[i]=='lower') else False for i in range(len(x_limit))])
    ax.errorbar(x_data[y_limited_low], y_data[y_limited_low], lolims = True, yerr=(np.ones(len(y_limited_low))*0.1)[y_limited_low], xerr=x_err[y_limited_low], fmt='o', color=cc, ecolor=cc, elinewidth=1, capsize=1)
    y_limited_up = np.array([True if (x_limit[i]=='nan' and y_limit[i]=='upper') else False for i in range(len(x_limit))])
    ax.errorbar(x_data[y_limited_up], y_data[y_limited_up], uplims = True, yerr=(np.ones(len(y_limited_up))*0.1)[y_limited_up], xerr=x_err[y_limited_up], fmt='o', color=cc, ecolor=cc, elinewidth=1, capsize=1)

    x_limited_low = np.array([True if (x_limit[i]=='lower' and y_limit[i]=='nan') else False for i in range(len(x_limit))])
    ax.errorbar(x_data[x_limited_low], y_data[x_limited_low], xlolims = True, yerr=y_err[x_limited_low], xerr = (np.ones(len(x_limited_low))*0.1)[x_limited_low], fmt='o', color=cc, ecolor=cc, elinewidth=1, capsize=1)
    x_limited_up = np.array([True if (x_limit[i]=='upper' and y_limit[i]=='nan') else False for i in range(len(x_limit))])
    ax.errorbar(x_data[x_limited_up], y_data[x_limited_up], xuplims = True, yerr=y_err[x_limited_up], xerr = (np.ones(len(x_limited_up))*0.1)[x_limited_up], fmt='o', color=cc, ecolor=cc, elinewidth=1, capsize=1)

    both_limited = np.array([True if (x_limit[i]=='lower' and y_limit[i]=='lower') else False for i in range(len(x_limit))])
    ax.errorbar(x_data[both_limited], y_data[both_limited], xlolims = True, lolims = True, xerr = (np.ones(len(both_limited))*0.1)[both_limited], yerr = (np.ones(len(both_limited))*0.05)[both_limited], fmt = 'o', color=cc, ecolor=cc, elinewidth=1, capsize=1)
##########


fig, ax = plt.subplots(2, 2, figsize=(10, 10))
figs, axs = plt.subplots(2, 2, figsize=(10, 10))
ax = ax.flatten()
axs = axs.flatten()

for aa in [ax, axs]:
    plot_Sodini(CFe_Sodini, OFe_Sodini, CFe_err_Sodini, OFe_err_Sodini, CFe_limit_type, OFe_limit_type, aa[0])
    aa[0].set_xlabel(r'[C/Fe]')
    aa[0].set_ylabel(r'[O/Fe]')

    plot_Sodini(CFe_Sodini, SiC_Sodini, CFe_err_Sodini, SiC_err_Sodini, CFe_limit_type, SiC_limit_type, aa[1])
    aa[1].set_xlabel(r'[C/Fe]')
    aa[1].set_ylabel(r'[Si/C]')

    plot_Sodini(OFe_Sodini, SiFe_Sodini, OFe_err_Sodini, SiFe_err_Sodini, OFe_limit_type, SiFe_limit_type, aa[2])
    aa[2].set_xlabel(r'[O/Fe]')
    aa[2].set_ylabel(r'[Si/Fe]')

    plot_Sodini(CO_Sodini, SiO_Sodini, CO_err_Sodini, SiO_err_Sodini, CO_limit_type, SiO_limit_type, aa[3])
    aa[3].set_xlabel(r'[C/O]')
    aa[3].set_ylabel(r'[Si/O]')


#gal = ['g5229300','g2274036', 'g519761','g137030','g37591','g33206', 'g10304', 'g5760', 'g1163', 'g578', 'g205', 'g39']
g = 'g33206'
run = ['test', 'test_newYields'] #['fiducial', 'no_ref/rad_map', 'com_point5/rad_map', 'com_point3/rad_map']
colours = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']
run_label = ['old Yields', 'new Yields'] #['fiducial', 'no refinement', 'ref 0.5 ckpc/h', 'ref 0.3 ckpc/h']

for r in run:  # Example: only plot the first 3 galaxies
    targethalo = TargetHalo(g, r)
    targethalo.read_haloes(snap, 0)
    halo_mass = targethalo.data[snap]['mass200']*1e10/lib.h
    coords, volume, redshift, _, _, _, h_density, hi_density, carbon_density, oxygen_density, silicon_density, iron_density = targethalo.gas_properties(snap, cond_hr, 1., all=False, metals=True)

    print('Gas properties collected.')

    _, star_initial_mass, _, star_birthday, star_mass = targethalo.stars_within_radius(snap, 1.)

    print('Star properties collected.')

    cond_10 = lib.time(star_birthday) > (lib.time(1/(1+targethalo.data[snap]['redshift'])) - 0.01)
    cond_100 = lib.time(star_birthday) > (lib.time(1/(1+targethalo.data[snap]['redshift'])) - 0.1)
    sfr_10 = np.sum(star_mass[cond_10]*1e10/lib.h)/1e7
    sfr_100 = np.sum(star_mass[cond_100]*1e10/lib.h)/1e8
    
    stellar_mass = np.sum(star_mass*1e10/lib.h)

    neutral_oxygen_density = oxygen_density*hi_density/h_density

    si_o = np.log10(silicon_density/oxygen_density) - Si_O_solar
    c_o = np.log10(carbon_density/oxygen_density) - C_O_solar
    c_fe = np.log10(carbon_density/iron_density) - C_Fe_solar
    o_fe = np.log10(oxygen_density/iron_density) - O_Fe_solar
    si_fe = np.log10(silicon_density/iron_density) - Si_Fe_solar
    si_c = np.log10(silicon_density/carbon_density) - Si_C_solar


    ax[0].scatter([], [], s=7, label = r'%s, $M_h = %.2e$ M$_\odot$, $M_* = %.2e$ M$_\odot$, SFR$_{10} = %.2f$ M$_\odot$ yr$^{-1}$, SFR$_{100} = %.2f$ M$_\odot$ yr$^{-1}$' % (run_label[run.index(r)], halo_mass, stellar_mass, sfr_10, sfr_100), color = colours[run.index(r)])
    axs[0].scatter([], [], s=7, label = r'%s, $M_h = %.2e$ M$_\odot$, $M_* = %.2e$ M$_\odot$, SFR$_{10} = %.2f$ M$_\odot$ yr$^{-1}$, SFR$_{100} = %.2f$ M$_\odot$ yr$^{-1}$' % (run_label[run.index(r)], halo_mass, stellar_mass, sfr_10, sfr_100), color = colours[run.index(r)])
    
    ax[0].scatter(c_fe, o_fe, s=0.5, alpha=0.1, color = colours[run.index(r)])
    add_contours(ax[0], c_fe, o_fe, colours[run.index(r)])
    ax[1].scatter(c_fe, si_c, s=0.5, alpha=0.1, color = colours[run.index(r)])
    add_contours(ax[1], c_fe, si_c, colours[run.index(r)])
    ax[2].scatter(o_fe, si_fe, s=0.5, alpha=0.1, color = colours[run.index(r)])
    add_contours(ax[2], o_fe, si_fe, colours[run.index(r)])
    ax[3].scatter(c_o, si_o, s=0.5, alpha=0.1, color = colours[run.index(r)])
    add_contours(ax[3], c_o, si_o, colours[run.index(r)])

#put the legend in a separate box below the 4 plots
handles, labels = ax[0].get_legend_handles_labels()
handless, labelss = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8)
figs.legend(handless, labelss, loc='lower center', ncol=2, fontsize=8)
fig.subplots_adjust(bottom=0.1)
fig.subplots_adjust(wspace=0.3, hspace=0.3)
figs.subplots_adjust(bottom=0.1)
figs.subplots_adjust(wspace=0.3, hspace=0.3)

fig.savefig('/home/gpruto/CGM_ref_analysis/images/metals/test_CGM_only_z%.1f_g33206_yieldscomp.png' % targethalo.data[snap]['redshift'], bbox_inches='tight')
#figs.savefig('/home/gpruto/CGM_ref_analysis/images/metals/test_CGM_only_z%.1f_g33206_contours_yieldscomp.png' % targethalo.data[snap]['redshift'], bbox_inches='tight')




