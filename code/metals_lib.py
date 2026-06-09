import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Polygon

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

def hist_2d(x, y, ax, x_bins=50, y_bins=50, color='Greys'):
    # Remove NaNs if needed
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    # 2D histogram
    h, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])

    # Plot with pcolormesh
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    ax.pcolormesh(X, Y, np.log10(h.T), shading='auto', cmap=color)



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

    ax.contour(xx, yy, f, levels=levels, colors=color, linewidths=1.)

##### PLOTTING OBSERVATIONAL DATA #####
def plot_Sodini(x_data, y_data, x_err, y_err, x_limit, y_limit, ax, color, special=False, label = None):
    
    all_constrained = np.array([True if (x_limit[i]=='nan' and y_limit[i]=='nan') else False for i in range(len(x_limit))])
    ax.errorbar(x_data[all_constrained], y_data[all_constrained], xerr=x_err[all_constrained], yerr=y_err[all_constrained], fmt='o', color=color, ecolor=color, elinewidth=1, capsize=1, label=label)

    y_limited_low = np.array([True if (x_limit[i]=='nan' and y_limit[i]=='lower') else False for i in range(len(x_limit))])
    ax.errorbar(x_data[y_limited_low], y_data[y_limited_low], lolims = True, yerr=(np.ones(len(y_limited_low))*0.3)[y_limited_low], xerr=x_err[y_limited_low], fmt='o', color=color, ecolor=color, elinewidth=1, capsize=1)
    y_limited_up = np.array([True if (x_limit[i]=='nan' and y_limit[i]=='upper') else False for i in range(len(x_limit))])
    ax.errorbar(x_data[y_limited_up], y_data[y_limited_up], uplims = True, yerr=(np.ones(len(y_limited_up))*0.3)[y_limited_up], xerr=x_err[y_limited_up], fmt='o', color=color, ecolor=color, elinewidth=1, capsize=1)

    x_limited_low = np.array([True if (x_limit[i]=='lower' and y_limit[i]=='nan') else False for i in range(len(x_limit))])
    ax.errorbar(x_data[x_limited_low], y_data[x_limited_low], xlolims = True, yerr=y_err[x_limited_low], xerr = (np.ones(len(x_limited_low))*0.3)[x_limited_low], fmt='o', color=color, ecolor=color, elinewidth=1, capsize=1)
    x_limited_up = np.array([True if (x_limit[i]=='upper' and y_limit[i]=='nan') else False for i in range(len(x_limit))])
    ax.errorbar(x_data[x_limited_up], y_data[x_limited_up], xuplims = True, yerr=y_err[x_limited_up], xerr = (np.ones(len(x_limited_up))*0.3)[x_limited_up], fmt='o', color=color, ecolor=color, elinewidth=1, capsize=1)

    both_limited_low = np.array([True if (x_limit[i]=='lower' and y_limit[i]=='lower') else False for i in range(len(x_limit))])
    ax.errorbar(x_data[both_limited_low], y_data[both_limited_low], xlolims = True, lolims = True, xerr = (np.ones(len(both_limited_low))*0.3)[both_limited_low], yerr = (np.ones(len(both_limited_low))*0.3)[both_limited_low], fmt = 'o', color=color, ecolor=color, elinewidth=1, capsize=1)

    x_low_y_up = np.array([True if (x_limit[i]=='lower' and y_limit[i]=='upper') else False for i in range(len(x_limit))])
    ax.errorbar(x_data[x_low_y_up], y_data[x_low_y_up], xlolims = True, uplims = True, xerr = (np.ones(len(x_low_y_up))*0.3)[x_low_y_up], yerr = (np.ones(len(x_low_y_up))*0.3)[x_low_y_up], fmt = 'o', color=color, ecolor=color, elinewidth=1, capsize=1)

    x_up_y_up = np.array([True if (x_limit[i]=='upper' and y_limit[i]=='upper') else False for i in range(len(x_limit))])
    ax.errorbar(x_data[x_up_y_up], y_data[x_up_y_up], xuplims = True, uplims = True, xerr = (np.ones(len(x_up_y_up))*0.3)[x_up_y_up], yerr = (np.ones(len(x_up_y_up))*0.3)[x_up_y_up], fmt = 'o', color=color, ecolor=color, elinewidth=1, capsize=1)

    if special==True:
        CO_Sodini_special = np.nan_to_num(CO_Sodini, copy=True, nan=-np.inf)
        maybepop3 = np.argsort(CO_Sodini_special)[-3:]
        ax.scatter(x_data[maybepop3], y_data[maybepop3], s=50, facecolors='none', edgecolors='black', marker='s', zorder=1e5)

##########


#### OBS DATA ####

# --- REDSHIFT ---
z_Sodini = np.array([5.44, 5.6268, 5.4869, 5.7763, 5.8385, 5.5624, 5.7912, 5.8767, 5.9917, 
                     5.9388, 5.8986, 5.9918, 6.0172, 5.6993, 5.8677, 6.1208, 6.1263, 5.7974, 
                     5.9450, 6.1114, 6.1434, 5.8990, 6.0642, 6.4041, 6.2743, 6.1228, 6.0611, 6.3784])
z_Becker = np.array([4.7393, 5.0817, 5.3380, 6.0115, 6.1312, 6.1988, 6.2575])
z_Poudel = np.array([4.987, 4.859, 5.050, 4.600, 4.627])
z_Poudel_2018 = np.array([5.335, 4.809, 4.829])


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

CFe_Becker = np.array([-0.12, 0.14, 0.13, 0, -0.09, np.nan, 0.13])
CFe_Becker_err = np.array([np.nan, np.nan, 0.06, 0.25, 0.23, np.nan, np.nan])
#CFe_Becker_lowlim = np.array([True if np.isnan(CFe_Becker_err[i]) else False for i in range(len(CFe_Becker_err))])
CFe_Becker_limit_type = np.array([np.nan if not np.isnan(CFe_Becker_err[i]) else 'lower' for i in range(len(CFe_Becker_err))])

FeH_Poudel = np.array([np.nan, -2.64, -1.53, -1.79, np.nan])
FeH_Poudel_err = np.array([np.nan, 0.2, 0.17, 0.33, np.nan])
CH_Poudel = np.array([-2.66, -2.44, -1.6, -2, -1.47])
CH_Poudel_err = np.array([0.19, 0.15, 0.19, np.nan, np.nan])
CFe_Poudel = CH_Poudel - FeH_Poudel
CFe_Poudel_err = np.sqrt(CH_Poudel_err**2 + FeH_Poudel_err**2)
CFe_Poudel_limit_type = np.array([np.nan if not np.isnan(CFe_Poudel_err[i]) else 'lower' for i in range(len(CFe_Poudel_err))])

CFe_Poudel_2018 = np.array([np.nan, -0.27, 0.13])
CFe_Poudel_2018_err = np.array([np.nan, 0.20, 0.20]) #not really sure, mainly based on image in Sodini
CFe_Poudel_2018_limit_type = np.array(['nan', 'nan', 'nan'])

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

OFe_Becker = np.array([0.02, 00.2, 0.55, 0.25, 0.56, np.nan, 0.21])
OFe_Becker_err = np.array([np.nan, np.nan, 0.06, 0.25, 0.29, np.nan, np.nan])
OFe_Becker_limit_type = np.array([np.nan if not np.isnan(OFe_Becker_err[i]) else 'lower' for i in range(len(OFe_Becker_err))])

OH_Poudel = np.array([-2.69, -2.14, -1.6, -1.46, -1.47])
OH_Poudel_err = np.array([0.17, np.nan, 0.18, np.nan, np.nan])
OFe_Poudel = OH_Poudel - FeH_Poudel
OFe_Poudel_err = np.sqrt(OH_Poudel_err**2 + FeH_Poudel_err**2)
OFe_Poudel_limit_type = np.array([np.nan if not np.isnan(OFe_Poudel_err[i]) else 'lower' for i in range(len(OFe_Poudel_err))])

OFe_Poudel_2018 = np.array([np.nan, -0.03, 0.43])
OFe_Poudel_2018_err = np.array([np.nan, 0.18, 0.2])
OFe_Poudel_2018_limit_type = np.array(['nan', 'nan', 'nan'])

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

SiFe_lowlim = np.array([True if np.isnan(SiFe_err_Sodini[i]) else False for i in range(len(SiFe_err_Sodini))])
SiFe_limit_type = np.array([np.nan if not np.isnan(SiFe_err_Sodini[i]) else 'lower' for i in range(len(SiFe_err_Sodini))])
SiFe_limit_type[np.where(SiFe_limit_type=='lower')[0][0]] = 'upper'

SiFe_Becker = np.array([0.35, 0.47, 0.42, 0.29, 0.25, np.nan, 0.18])
SiFe_Becker_err = np.array([0.06, 0.07, 0.05, 0.25, 0.19, np.nan, np.nan])
SiFe_Becker_lowlim = np.array([True if np.isnan(SiFe_Becker_err[i]) else False for i in range(len(SiFe_Becker_err))])
SiFe_Becker_limit_type = np.array([np.nan if not np.isnan(SiFe_Becker_err[i]) else 'lower' for i in range(len(SiFe_Becker_err))])

SiH_Poudel = np.array([-1.89, -1.91, -1.31, -1.29, -1])
SiH_Poudel_err = np.array([0.16, 0.18, 0.18, 0.16, np.nan])
SiFe_Poudel = SiH_Poudel - FeH_Poudel
#SiFe_Poudel_err = np.sqrt(SiH_Poudel_err**2 + FeH_Poudel_err**2)
SiFe_Poudel_err = np.array([np.nan, 0.20])
SiFe_Poudel_limit_type = np.array([np.nan if not np.isnan(SiFe_Poudel_err[i]) else 'lower' for i in range(len(SiFe_Poudel_err))])

SiFe_Poudel_2018 = np.array([np.nan, 0.17, 0.39])
SiFe_Poudel_2018_err = np.array([np.nan, 0.22, 0.15])
SiFe_Poudel_2018_limit_type = np.array(['nan', 'nan', 'nan'])

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

CO_Becker = np.array([14.6-15 - C_O_solar, 14.3 - 14.7 - C_O_solar, -0.42, -0.25, -0.65, -0.34, -0.08])
CO_Becker_err = np.array([np.nan, np.nan, 0.07, 0.06, 0.27, 0.17, 0.26])
CO_Becker_limit_type = np.array([np.nan if not np.isnan(CO_Becker_err[i]) else 'lower' for i in range(len(CO_Becker_err))])
#CO_Becker_lowlim = np.array([True if np.isnan(CO_Becker_err[i]) else False for i in range(len(CO_Becker_err))])

CO_Poudel = np.array([0.03, -0.3, 0.00, np.nan, np.nan])
CO_Poudel_err = np.array([0.15, np.nan, 0.15, np.nan, np.nan])
CO_Poudel_limit_type = np.array([np.nan if not np.isnan(CO_Poudel_err[i]) else 'upper' for i in range(len(CO_Poudel_err))])

CO_Poudel_2018 = np.array([-0.11, -0.24, -0.3])
CO_Poudel_2018_err = np.array([0.07, 0.1, 0.19])
CO_Poudel_2018_limit_type = np.array(['nan', 'nan', 'nan'])

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


SiO_Becker = np.array([0.33, 0.27, -0.14, 0.04, -0.32, -0.2, -0.02])
SiO_Becker_err = np.array([np.nan, np.nan, 0.06, 0.04, 0.24, 0.14, 0.17])
SiO_Becker_limit_type = np.array([np.nan if not np.isnan(SiO_Becker_err[i]) else 'upper' for i in range(len(SiO_Becker_err))])


SiO_Poudel = np.array([0.79, 0.23, 0.3, 0.17, np.nan])
SiO_Poudel_err = np.array([0.09, np.nan, 0.14, np.nan, np.nan])
SiO_Poudel_limit_type = np.array([np.nan if not np.isnan(SiO_Poudel_err[i]) else 'upper' for i in range(len(SiO_Poudel_err))])

SiO_Poudel_2018 = np.array([0.02, 0.20, -0.04])
SiO_Poudel_2018_err = np.array([0.06, 0.15, 0.17])
SiO_Poudel_2018_limit_type = np.array(['nan', 'nan', 'nan'])

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

FeO_Becker = -OFe_Becker
FeO_Becker_err = OFe_Becker_err
FeO_Becker_limit_type = np.array([np.nan if not np.isnan(OFe_Becker_err[i]) else 'upper' for i in range(len(OFe_Becker_err))])

FeO_Poudel = np.array([np.nan, -0.50, 0.07, -0.33, np.nan])
FeO_Poudel_err = np.array([np.nan, np.nan, 0.14, np.nan, np.nan])
FeO_Poudel_limit_type = np.array([np.nan if not np.isnan(FeO_Poudel_err[i]) else 'upper' for i in range(len(FeO_Poudel_err))])

FeO_Poudel_2018 = np.array([np.nan, 0.03, -0.43])
FeO_Poudel_2018_err = np.array([np.nan, 0.18, 0.20])
FeO_Poudel_2018_limit_type = np.array(['nan', 'nan', 'nan'])

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

SiC_Becker = np.array([0.46, 0.33, 0.29, 0.29, 0.34, 0.14, 0.06])
SiC_Becker_err = np.array([np.nan, np.nan, 0.06, 0.06, 0.16, 0.12, 0.23])
SiC_Becker_limit_type = np.array(['nan' if not np.isnan(SiC_Becker_err[i]) else 'upper' for i in range(len(SiC_Becker_err))])

SiC_Poudel = SiH_Poudel - CH_Poudel
SiC_Poudel_err = np.array([np.nan, 0.20])
SiC_Poudel_limit_type = np.array(['nan' if not np.isnan(SiC_Poudel_err[i]) else 'upper' for i in range(len(SiC_Poudel_err))])

SiC_Poudel_2018 = np.array([0.13, 0.44, 0.26])
SiC_Poudel_2018_err = np.array([0.06, 0.16, 0.20])
SiC_Poudel_2018_limit_type = np.array(['nan', 'nan', 'nan'])
###################

# --- Fe/H ---
FeH_Sodini = np.array([np.nan, np.nan, np.nan, -2.75, -2.25, np.nan, np.nan, -3.21, -3.33, -3.18,
                       np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                         -2.78, np.nan, np.nan, -2.74, np.nan, np.nan, np.nan, np.nan ])

FeH_err_Sodini = np.array([np.nan, np.nan, np.nan, 0.11, 0.11, np.nan, np.nan, 0.13, 0.22, 0.12,
                           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                           0.10, np.nan, np.nan, 0.10, np.nan, np.nan, np.nan, np.nan])

FeH_err_limit_type = np.array(['nan' if not np.isnan(FeH_err_Sodini[i]) else 'upper' for i in range(len(FeH_err_Sodini))])


########POP II models###########



### Woosley & Weaver 1995
CFe0_WW = [-0.14364640883977908, -0.29558011049723765, -0.3646408839779006, -0.32320441988950277, -0.15055248618784534, -0.15055248618784534, 0.23618784530386727, 0.8646408839779005, 1.0787292817679557, 1.0787292817679557, 0.6022099447513811, 0.20165745856353579, 0.2154696132596685, 0.04281767955801108, -0.03314917127071826]
OFe0_WW = [-0.09230769230769231, 0.05384615384615388, 0.24615384615384617, 0.4538461538461539, 0.6076923076923078, 0.676923076923077, 1.0615384615384615, 1.6923076923076925, 1.4923076923076923, 1.3153846153846156, 0.8076923076923077, 0.41538461538461546, 0.24615384615384617, 0.03076923076923077, 0.05384615384615388]

CFe1_WW = [-0.25619834710743805, -0.18044077134986225, -0.07713498622589532, 0.5082644628099173, 0.6253443526170799, 0.8526170798898072, 1.0798898071625347, 0.9696969696969697, 0.859504132231405, 0.522038567493113, 0.3980716253443526, 0.2603305785123968, 0.16391184573002748, -0.03581267217630857, -0.20798898071625338]
SiC1_WW = [0.3076923076923077, 0.41666666666666663, 0.5897435897435896, 0.5897435897435896, 0.46153846153846145, 0.45512820512820507, 0.17307692307692302, 0.03205128205128205, 0.16666666666666663, 0.16025641025641024, 0.3012820512820512, 0.3012820512820512, 0.16666666666666663, 0.16025641025641024, 0.21794871794871795]

OFe2_WW = [1.3232044198895028, 1.5787292817679557, 1.8273480662983426, 1.5925414364640882, 1.5787292817679557, 1.0469613259668509, 0.20441988950276235, -0.0787292817679558, -0.07182320441988954, 0.43232044198895025]
SiFe2_WW = [1.3397435897435896, 1.314102564102564, 1.1538461538461537, 0.9871794871794871, 0.9615384615384615, 0.5897435897435898, 0.01282051282051282, 0.20512820512820512, 0.3974358974358974, 0.75]

CO3_WW = [-0.5966850828729282, -0.5469613259668509, -0.4640883977900554, -0.40607734806629847, -0.40607734806629847, -0.33149171270718236, -0.2320441988950277, -0.26519337016574585, -0.3646408839779005, -0.41436464088397784, -0.5552486187845305, -0.6712707182320442, -0.6629834254143647]
SiO3_WW = [-0.558641975308642, -0.43518518518518534, -0.4197530864197532, -0.2885802469135803, -0.18055555555555558, -0.1728395061728396, 0.1049382716049383, 0.20524691358024683, 0.2129629629629628, 0.08179012345678993, 0.08179012345678993, -0.1728395061728396, -0.4429012345679013]


def plot_models(x_data, y_data, ax, color, label=None):
    #create a polygon with x_data and y_data
    points = list(zip(x_data, y_data))
    poly = Polygon(points, closed=True, facecolor=color, edgecolor=color, alpha=0.4, label=label)
    ax.add_patch(poly)


####### tracks #######

Portinari_metallicities = [0.0004, 0.004, 0.008, 0.02, 0.05]

Portinari_masses = [9, 12, 15, 20, 30, 40, 60, 100, 120]

Portinari_elements = ["1H","3He","4He","12C","13C","14N","15N","16O","17O","18O", "20Ne","22Ne","24Mg","28Si","32S","40Ca","56Fe"]

Portinari_yields = np.zeros((5, 9, len(Portinari_elements))) #shape is (metallicity, mass, element)

### Z = 0.0004
Portinari_yields[0,0,:] = [4.57, 1.22E-04, 2.63, 8.97E-02, 2.27E-05, 9.31E-04, 7.16E-06, 0.140, 4.78E-06, 2.43E-05, 
                           9.43E-03, 2.96E-05, 5.61E-03, 3.08E-02, 1.38E-02, 2.60E-03, 0.155]

Portinari_yields[0,1,:] = [5.77, 9.50E-05, 3.45, 0.170, 2.91E-05, 1.28E-03, 3.58E-05, 0.671, 5.09E-06, 2.97E-05,
                           0.108, 3.82E-05, 2.91E-02, 8.63E-02, 4.39E-02, 7.91E-03, 0.157]

Portinari_yields[0,2,:] = [6.86, 8.02E-05, 4.16, 0.214, 3.58E-05, 1.50E-03, 7.45E-05, 1.41, 5.80E-06, 7.57E-05,
                           9.92E-02, 6.44E-05, 2.68E-02, 0.200, 0.118, 1.57E-02, 9.59E-02]

Portinari_yields[0,3,:] = [8.51, 5.83E-05, 5.24, 0.284, 4.21E-05, 2.18E-03, 1.29E-04, 3.03, 5.66E-06, 3.01E-06,
                           9.99E-02, 5.64E-05, 7.59E-02, 0.340, 0.172, 1.63E-02, 0.157]

Portinari_yields[0,4,:] = [11.54, 2.59E-05, 7.22, 0.350, 4.49E-05, 3.33E-03, 6.14E-05, 3.15, 1.03E-05, 1.34E-04,
                           0.616, 2.25E-04, 0.130, 1.21E-02, 2.90E-04, 2.42E-05, 4.39E-04]

Portinari_yields[0,5,:] = [14.33, 1.73E-05, 9.26, 0.339, 5.00E-05, 4.86E-03, 6.15E-07, 0.824, 1.71E-05, 3.60E-06,
                           0.271, 9.69E-05, 5.43E-02, 3.08E-04, 1.87E-04, 2.83E-05, 5.52E-04]

Portinari_yields[0,6,:] = [20.01, 1.09E-05, 14.60, 0.510, 8.04E-05, 7.57E-03, 8.86E-07, 0.288, 3.08E-05, 3.31E-06,
                           1.14E-03, 4.64E-04, 3.34E-04, 4.63E-04, 2.81E-04, 4.24E-05, 8.29E-04]

Portinari_yields[0,7,:] = [28.07, 5.20E-05, 23.46, 0.719, 2.20E-04, 1.08E-02, 1.21E-06, 13.95, 1.52E-05, 7.60E-06,
                           1.00, 2.12E-04, 0.718, 6.73E-03, 4.08E-03, 6.18E-04, 1.21E-03]

Portinari_yields[0,8,:] = [32.80, 5.05E-05, 33.25, 1.97, 1.77E-04, 1.58E-02, 1.86E-06, 19.16, 5.69E-05, 8.80E-06,
                           1.33, 3.03E-04, 0.867, 9.21E-04, 5.58E-04, 8.44E-05, 1.65E-03]

#### Z = 0.004 - it lacks M=60
Portinari_yields[1,0,:] = [4.44, 1.40E-04, 2.71, 9.03E-02, 2.49E-04, 8.60E-03, 1.45E-05, 0.162, 4.00E-05, 2.19E-05,
                           1.54E-02, 2.94E-04, 6.65E-03, 3.74E-02, 1.90E-02, 3.71E-03, 0.169]
Portinari_yields[1,1,:] = [5.59, 1.15E-04, 3.54, 0.171, 3.23E-04, 1.16E-02, 5.92E-05, 0.707, 6.26E-05, 4.81E-04,
                           0.121, 3.86E-04, 3.92E-02, 9.22E-02, 4.08E-02, 6.86E-03, 0.169]
Portinari_yields[1,2,:] = [6.71, 1.00E-04, 4.28, 0.240, 3.73E-04, 1.52E-02, 7.12E-05, 1.31, 8.36E-05, 2.95E-05,
                           0.141, 4.53E-04, 5.86E-02, 0.194, 9.69E-02, 1.40E-02, 0.125]
Portinari_yields[1,3,:] = [8.40, 1.07E-04, 5.38, 0.271, 4.89E-04, 1.87E-02, 1.33E-04, 2.95, 7.66E-05, 3.98E-05,
                           9.31E-02, 5.68E-04, 4.77E-02, 0.374, 0.200, 2.02E-02, 0.192]

Portinari_yields[1,4,:] = [11.39, 8.51E-05, 7.58, 0.319, 4.35E-04, 3.13E-02, 5.96E-05, 3.03, 3.20E-04, 4.12E-05,
                           0.510, 7.82E-04, 0.131, 2.32E-02, 2.38E-03, 2.77E-04, 4.46E-03]
Portinari_yields[1,5,:] = [14.41, 7.77E-05, 9.46, 0.361, 5.72E-04, 4.06E-02, 8.17E-06, 2.12, 3.60E-04, 3.31E-05,
                           0.743, 9.84E-04, 0.223, 3.13E-03, 1.90E-03, 2.87E-04, 5.61E-03]
Portinari_yields[1,7,:] = [28.17, 1.26E-04, 26.66, 0.375, 9.20E-04, 0.124, 1.52E-05, 6.98, 4.04E-04, 6.04E-05,
                           0.515, 2.26E-03, 0.379, 7.19E-03, 4.36E-03, 6.60E-04, 1.29E-02]
Portinari_yields[1,8,:] = [32.05, 1.20E-04, 52.10, 0.127, 1.63E-03, 0.224, 2.04E-05, 0.147, 3.15E-04, 6.12E-05,
                           2.59E-02, 3.79E-03, 8.00E-03, 1.11E-02, 6.71E-03, 1.01E-03, 1.98E-02]

#### Z = 0.008 - it lacks M = 100

Portinari_yields[2,0,:] = [4.40, 1.60E-04, 2.52, 0.259, 5.06E-04, 1.47E-02, 2.21E-05, 0.220, 1.69E-04, 5.25E-04,
                           8.34E-03, 3.25E-03, 3.68E-03, 4.12E-02, 2.81E-02, 5.68E-03, 0.139 ]
Portinari_yields[2,1,:] = [5.50, 1.80E-04, 3.67, 0.168, 6.69E-04, 2.32E-02, 7.17E-05, 0.696, 2.28E-05, 1.91E-04,
                           0.119, 7.62E-04, 3.42E-02, 9.48E-02, 4.50E-02, 7.55E-03, 0.158 ]
Portinari_yields[2,2,:] = [6.49, 1.78E-04, 4.45, 0.248, 7.84E-04, 2.92E-02, 7.79E-05, 1.31, 2.79E-05, 6.39E-05,
                           0.143, 9.13E-04, 5.18E-02, 0.204, 9.92E-02, 1.33E-02, 0.120 ]
Portinari_yields[2,3,:] = [8.27, 1.55E-04, 5.59, 0.271, 1.03E-03, 3.67E-02, 1.39E-04, 2.85, 2.42E-04, 7.94E-05,
                           0.107, 1.15E-03, 4.42E-02, 0.372, 0.195, 2.01E-02, 0.194]
Portinari_yields[2,4,:] = [11.23, 1.14E-04, 7.75, 0.313, 8.34E-04, 5.79E-02, 6.87E-05, 3.06, 4.84E-04, 5.71E-05,
                           0.502, 1.57E-03, 0.133, 2.93E-02, 4.14E-03, 5.03E-04, 8.96E-03]
Portinari_yields[2,5,:] = [13.90, 2.19E-04, 10.19, 0.352, 1.02E-03, 7.69E-02, 3.36E-05, 2.21, 4.89E-04, 1.19E-04,
                           0.656, 1.99E-03, 6.74E-03, 6.35E-03, 4.04E-03, 6.25E-04, 1.14E-02]
Portinari_yields[2,6,:] = [18.91, 2.67E-04, 22.35, 7.33, 2.02E-03, 0.133, 1.04E-04, 4.78, 7.08E-04, 2.33E-03,
                           0.517, 9.59E-03, 0.184, 5.35E-02, 9.73E-03, 1.24E-03, 2.34E-02]
Portinari_yields[2,8,:] = [30.08, 3.51E-04, 49.35, 19.95, 4.85E-03, 0.349, 8.52E-05, 9.17, 8.51E-04, 1.93E-04,
                           0.638, 0.300, 8.17E-02, 3.34E-02, 1.75E-02, 2.60E-03, 4.99E-02]

#### Z = 0.02
Portinari_yields[3,0,:] = [4.18, 2.28E-04, 2.99, 7.27E-02, 1.29E-03, 4.02E-02, 4.85E-05, 0.178, 0.748E-04, 1.18E-04,
                           1.95E-02, 1.50E-03, 6.64E-03, 7.62E-02, 6.22E-02, 1.18E-02, 5.68E-02]
Portinari_yields[3,1,:] = [5.20, 2.45E-04, 3.95, 0.170, 1.65E-03, 5.46E-02, 1.12E-04, 0.712, 0.827E-04, 1.49E-04,
                           0.123, 1.92E-03, 2.78E-02, 0.111, 6.11E-02, 1.04E-02, 0.121]
Portinari_yields[3,2,:] = [6.18, 2.65E-04, 4.70, 0.266, 1.95E-03, 6.68E-02, 1.02E-04, 1.273, 0.882E-04, 1.66E-04,
                           0.175, 2.28E-03, 3.62E-02, 0.213, 9.46E-02, 1.03E-02, 9.79E-02]
Portinari_yields[3,3,:] = [7.7, 3.23E-04, 6.01, 0.279, 2.06E-03, 8.84E-02, 1.65E-04, 2.76, 1.04E-03, 1.99E-04,
                           0.214, 2.89E-03, 6.91E-02, 0.335, 0.156, 1.68E-02, 0.186]
Portinari_yields[3,4,:] = [10.30, 3.96E-04, 8.28, 0.307, 2.01E-03, 0.128, 9.46E-05, 3.08, 1.45E-03, 2.32E-04,
                           0.500, 3.89E-03, 0.124, 4.07E-02, 9.02E-03, 1.17E-03, 2.22E-02]
Portinari_yields[3,5,:] = [12.52, 4.72E-04, 19.05, 3.03, 3.61E-03, 0.211, 1.35E-04, 2.05, 3.37E-03, 7.00E-03,
                           0.135, 0.282, 3.86E-02, 0.298, 0.154, 1.53E-02, 0.133]
Portinari_yields[3,6,:] = [16.49, 5.27E-04, 32.14, 4.93, 5.41E-03, 0.343, 1.81E-04, 2.66, 3.84E-03, 2.89E-04,
                           0.138, 0.336, 5.60E-02, 0.360, 0.181, 1.87E-02, 0.234]
Portinari_yields[3,7,:] = [23.97, 5.51E-04, 55.48, 11.40, 9.79E-03, 0.682, 2.55E-04, 4.27, 5.58E-03, 3.04E-04,
                           0.455, 0.613, 0.130, 0.371, 0.176, 2.13E-02, 0.251 ]
Portinari_yields[3,8,:] = [28.72, 1.37E-03, 74.03, 8.63, 1.16E-02, 1.01, 3.25E-04, 3.52, 2.17E-03, 7.53E-04,
                           0.349, 0.542, 0.113, 0.402, 0.197, 2.28E-02, 0.307]

#### Z = 0.05 
Portinari_yields[4,0,:] = [3.45, 4.06E-04, 3.45, 0.111, 3.19E-03, 8.79E-02, 7.73E-05, 0.283, 2.52E-03, 2.14E-03,
                           3.49E-02, 3.74E-03, 1.16E-02, 9.26E-02, 7.55E-02, 1.42E-02, 6.61E-02]
Portinari_yields[4,1,:] = [4.25, 4.92E-04, 4.46, 0.225, 3.81E-03, 0.116, 1.44E-04, 0.895, 2.77E-03, 5.47E-03,
                           0.168, 4.73E-03, 3.83E-02, 0.124, 6.51E-02, 1.05E-02, 0.129]
Portinari_yields[4,2,:] = [5.02, 5.74E-04, 5.22, 0.329, 4.79E-03, 0.136, 1.30E-04, 1.57, 3.37E-03, 8.71E-03,
                           0.123, 5.63E-03, 3.05E-02, 0.285, 0.135, 1.43E-02, 0.131]
Portinari_yields[4,3,:] = [6.22, 7.36E-04, 6.24, 0.376, 4.79E-03, 0.174, 2.34E-04, 3.44, 4.24E-03, 1.79E-03,
                           0.412, 6.75E-03, 0.124, 0.299, 0.132, 1.57E-02, 0.152]
Portinari_yields[4,5,:] = [9.98, 1.17E-03, 22.79, 2.34, 7.69E-03, 0.484, 2.43E-04, 1.32, 1.40E-02, 7.61E-03,
                           0.301, 0.646, 7.56E-02, 0.173, 9.11E-02, 1.40E-02, 0.203]
Portinari_yields[4,6,:] = [14.06, 6.47E-04, 36.03, 3.66, 1.14E-02, 1.06, 2.19E-04, 1.73, 3.39E-02, 3.67E-04,
                           0.448, 0.562, 0.111, 0.217, 0.106, 1.50E-02, 0.238]
Portinari_yields[4,7,:] = [26.74, 8.60E-04, 60.90, 3.79, 1.94E-02, 1.79, 2.78E-04, 2.14, 5.52E-02, 3.23E-02,
                        0.594, 1.27, 0.157, 0.281, 0.146, 2.12E-02, 0.357]
Portinari_yields[4,8,:] = [38.07, 1.15E-03, 68.39, 3.82, 2.61E-02, 2.11, 3.20E-04, 2.60, 6.31E-02, 0.177,
                           0.669, 1.27, 0.180, 0.314, 0.166, 2.42E-02, 0.416]