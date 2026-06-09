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
import pandas as pd 

all = True

#from Asplund 2010
C_sun = 8.47
O_sun = 8.73
Si_sun = 7.55
Fe_sun = 7.54

Si_O_solar = Si_sun - O_sun
Si_Fe_solar = Si_sun - Fe_sun
Si_C_solar = Si_sun - C_sun
C_O_solar = C_sun - O_sun
C_Fe_solar = C_sun - Fe_sun
O_Fe_solar = O_sun - Fe_sun


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
def plot_Sodini(x_data, y_data, x_err, y_err, x_limit, y_limit, ax):
    all_constrained = np.array([True if (x_limit[i]=='nan' and y_limit[i]=='nan') else False for i in range(len(x_limit))])
    ax.errorbar(x_data[all_constrained], y_data[all_constrained], xerr=x_err[all_constrained], yerr=y_err[all_constrained], fmt='o', color='grey', ecolor='grey', elinewidth=1, capsize=1)

    y_limited_low = np.array([True if (x_limit[i]=='nan' and y_limit[i]=='lower') else False for i in range(len(x_limit))])
    ax.errorbar(x_data[y_limited_low], y_data[y_limited_low], lolims = True, yerr=(np.ones(len(y_limited_low))*0.1)[y_limited_low], xerr=x_err[y_limited_low], fmt='o', color='grey', ecolor='grey', elinewidth=1, capsize=1)
    y_limited_up = np.array([True if (x_limit[i]=='nan' and y_limit[i]=='upper') else False for i in range(len(x_limit))])
    ax.errorbar(x_data[y_limited_up], y_data[y_limited_up], uplims = True, yerr=(np.ones(len(y_limited_up))*0.1)[y_limited_up], xerr=x_err[y_limited_up], fmt='o', color='grey', ecolor='grey', elinewidth=1, capsize=1)

    x_limited_low = np.array([True if (x_limit[i]=='lower' and y_limit[i]=='nan') else False for i in range(len(x_limit))])
    ax.errorbar(x_data[x_limited_low], y_data[x_limited_low], xlolims = True, yerr=y_err[x_limited_low], xerr = (np.ones(len(x_limited_low))*0.1)[x_limited_low], fmt='o', color='grey', ecolor='grey', elinewidth=1, capsize=1)
    x_limited_up = np.array([True if (x_limit[i]=='upper' and y_limit[i]=='nan') else False for i in range(len(x_limit))])
    ax.errorbar(x_data[x_limited_up], y_data[x_limited_up], xuplims = True, yerr=y_err[x_limited_up], xerr = (np.ones(len(x_limited_up))*0.1)[x_limited_up], fmt='o', color='grey', ecolor='grey', elinewidth=1, capsize=1)

    both_limited = np.array([True if (x_limit[i]=='lower' and y_limit[i]=='lower') else False for i in range(len(x_limit))])
    ax.errorbar(x_data[both_limited], y_data[both_limited], xlolims = True, lolims = True, xerr = (np.ones(len(both_limited))*0.1)[both_limited], yerr = (np.ones(len(both_limited))*0.05)[both_limited], fmt = 'o', color='grey', ecolor='grey', elinewidth=1, capsize=1)
##########


if all==False:
    ####### ANALYSING THE CGM ONLY ######
    ref_levels = ['no_ref', 'com_point5', 'com_point3']
    rad = ['rad_map']#, 'no_rad', 'uvb_only', 'stars_only']
    sim = []
    for i in range(len(ref_levels)):
        for j in range(len(rad)):
            sim.append(f'{ref_levels[i]}/{rad[j]}')
    snap = 69
    gal = 'g33206'

    haloes = []
    mask = np.zeros(len(sim), dtype=bool)
    for i in range(len(sim)):
        haloes.append(TargetHalo(gal, sim[i]))
        if not os.path.isdir('/cephfs/gpruto/CGM_ref/run/%s/%s' %(gal, sim[i])):
            print(" We didn't simulate galaxy %s with %s" %(gal, sim[i]))
            continue
        if not os.path.isdir('/cephfs/gpruto/CGM_ref/run/%s/%s/output/groups_%03d' %(gal, sim[i], snap)):
            print("We don't have snapshot %03d for galaxy %s with %s" %(snap, gal, sim[i]))
            continue
        else:
            mask[i] = True
            haloes[i].read_haloes(snap, 0)
            print(haloes[i].data[snap]['radius'])

    Nplot = 200
    cd_OI = [[] for i in range(len(haloes))]
    cd_C = [[] for i in range(len(haloes))]
    cd_Si = [[] for i in range(len(haloes))]
    cd_Fe = [[] for i in range(len(haloes))]
    cd_O = [[] for i in range(len(haloes))]

    for i in range(len(sim)):
        if not mask[i]:
            continue
        Edges1d, density_OI = haloes[i].column_density_map(snap, 0.5, 1., Nplot, box=False, element='OI')
        _, density_C = haloes[i].column_density_map(snap, 0.5, 1., Nplot, box=False, element='C')
        _, density_Si = haloes[i].column_density_map(snap, 0.5, 1., Nplot, box=False, element='Si')
        _, density_Fe = haloes[i].column_density_map(snap, 0.5, 1., Nplot, box=False, element='Fe')
        _, density_O = haloes[i].column_density_map(snap, 0.5, 1., Nplot, box=False, element='O')

        cd_OI[i] = density_OI
        cd_C[i] = density_C
        cd_Si[i] = density_Si
        cd_Fe[i] = density_Fe
        cd_O[i] = density_O

        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax = ax.flatten()

        ax[0].scatter(np.log10(density_C/density_Fe) - C_Fe_solar, np.log10(density_O/density_Fe) - O_Fe_solar, color='tab:blue', s=5)
        plot_Sodini(CFe_Sodini, OFe_Sodini, CFe_err_Sodini, OFe_err_Sodini, CFe_limit_type, OFe_limit_type, ax[0])
        ax[0].set_xlabel(r'[C/Fe]')
        ax[0].set_ylabel(r'[O/Fe]')

        ax[1].scatter(np.log10(density_Si/density_C) - Si_C_solar, np.log10(density_Si/density_Fe) - Si_Fe_solar, color='tab:blue', s=5)
        plot_Sodini(CFe_Sodini, SiC_Sodini, CFe_err_Sodini, SiC_err_Sodini, CFe_limit_type, SiC_limit_type, ax[1])
        ax[1].set_xlabel(r'[C/Fe]')
        ax[1].set_ylabel(r'[Si/C]')

        ax[2].scatter(np.log10(density_O/density_Fe) - O_Fe_solar, np.log10(density_Si/density_Fe) - Si_Fe_solar, color='tab:blue', s=5)
        plot_Sodini(OFe_Sodini, SiFe_Sodini, OFe_err_Sodini, SiFe_err_Sodini, OFe_limit_type, SiFe_limit_type, ax[2])
        ax[2].set_xlabel(r'[O/Fe]')
        ax[2].set_ylabel(r'[Si/Fe]')

        ax[3].scatter(np.log10(density_C/density_O) - C_O_solar, np.log10(density_Si/density_O) - Si_O_solar, color='tab:blue', s=5)
        plot_Sodini(CO_Sodini, SiO_Sodini, CO_err_Sodini, SiO_err_Sodini, CO_limit_type, SiO_limit_type, ax[3])
        ax[3].set_xlabel(r'[C/O]')
        ax[3].set_ylabel(r'[Si/O]')


        fig.savefig('/home/gpruto/CGM_ref_analysis/images/metals/Sodini_comparison_CGM_only_%s_radmap.png' %(ref_levels[i]), dpi=300, bbox_inches ='tight')

else:
    filein = '/home/gpruto/CGM_ref_analysis/code/metals/cd_inside_g33206_noref_radmap_69_600x600.txt'
    data = pd.read_csv(filein, delim_whitespace=True)
    print(data.keys())
    fig, ax = plt.subplots(2,2, figsize=(10,10))
    ax = ax.flatten()

    observable_condition = data['OI_cd_z']>1e13
    data_obs = data[observable_condition]
    ax[0].scatter(np.log10(data['C_cd_z']/data['Fe_cd_z']) - C_Fe_solar, np.log10(data['O_cd_z']/data['Fe_cd_z']) - O_Fe_solar, color='black', s=2, label = 'All')
    ax[0].scatter(np.log10(data_obs['C_cd_z']/data_obs['Fe_cd_z']) - C_Fe_solar, np.log10(data_obs['O_cd_z']/data_obs['Fe_cd_z']) - O_Fe_solar, color='tab:blue', s=3, label = r'$N_{\rm OI} > 10^{13}$ cm$^{-2}$')
    plot_Sodini(CFe_Sodini, OFe_Sodini, CFe_err_Sodini, OFe_err_Sodini, CFe_limit_type, OFe_limit_type, ax[0])
    ax[0].set_xlabel(r'[C/Fe]')
    ax[0].set_ylabel(r'[O/Fe]')
    ax[0].legend()

    ax[1].scatter(np.log10(data['Si_cd_z']/data['C_cd_z']) - Si_C_solar, np.log10(data['Si_cd_z']/data['Fe_cd_z']) - Si_Fe_solar, color='black', s=2)
    ax[1].scatter(np.log10(data_obs['Si_cd_z']/data_obs['C_cd_z']) - Si_C_solar, np.log10(data_obs['Si_cd_z']/data_obs['Fe_cd_z']) - Si_Fe_solar, color='tab:blue', s=3)
    plot_Sodini(CFe_Sodini, SiC_Sodini, CFe_err_Sodini, SiC_err_Sodini, CFe_limit_type, SiC_limit_type, ax[1])
    ax[1].set_xlabel(r'[C/Fe]')
    ax[1].set_ylabel(r'[Si/C]')

    ax[2].scatter(np.log10(data['O_cd_z']/data['Fe_cd_z']) - O_Fe_solar, np.log10(data['Si_cd_z']/data['Fe_cd_z']) - Si_Fe_solar, color='black', s=2)
    ax[2].scatter(np.log10(data_obs['O_cd_z']/data_obs['Fe_cd_z']) - O_Fe_solar, np.log10(data_obs['Si_cd_z']/data_obs['Fe_cd_z']) - Si_Fe_solar, color='tab:blue', s=3)
    plot_Sodini(OFe_Sodini, SiFe_Sodini, OFe_err_Sodini, SiFe_err_Sodini, OFe_limit_type, SiFe_limit_type, ax[2])
    ax[2].set_xlabel(r'[O/Fe]')
    ax[2].set_ylabel(r'[Si/Fe]')

    ax[3].scatter(np.log10(data['C_cd_z']/data['O_cd_z']) - C_O_solar, np.log10(data['Si_cd_z']/data['O_cd_z']) - Si_O_solar, color='black', s=2)
    ax[3].scatter(np.log10(data_obs['C_cd_z']/data_obs['O_cd_z']) - C_O_solar, np.log10(data_obs['Si_cd_z']/data_obs['O_cd_z']) - Si_O_solar, color='tab:blue', s=3)
    plot_Sodini(CO_Sodini, SiO_Sodini, CO_err_Sodini, SiO_err_Sodini, CO_limit_type, SiO_limit_type, ax[3])
    ax[3].set_xlabel(r'[C/O]')
    ax[3].set_ylabel(r'[Si/O]')

    fig.savefig('/home/gpruto/CGM_ref_analysis/images/metals/Sodini_comparison_all_g33206_radmap_noref_z5.png', dpi=300, bbox_inches ='tight')


