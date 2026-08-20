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


red = int(sys.argv[1])
gal = ['g5229300', 'g2274036', 'g519761', 'g500531', 'g137030', 'g37591','g33206', 'g10304', 'g5760', 'g1163', 'g578', 'g205', 'g39', 'g2']
fig1 = False
fig3 = True
figpopIII = False


if fig3:
    #### PLOT ABUNDANCES TO COMPARE TO SODINI
    c_fe_bins = np.linspace(-1.5, 1.8, 300)
    o_fe_bins = np.linspace(-1.5, 2.5, 300)
    si_c_bins = np.linspace(-1.5, 1., 300)
    si_fe_bins = np.linspace(-0.5, 1.2, 300)
    si_o_bins = np.linspace(-1.5, 1.2, 300)
    c_o_bins = np.linspace(-1.2, 1.2, 300)

    hist1_total = np.zeros((len(c_fe_bins)-1, len(o_fe_bins)-1))
    hist2_total = np.zeros((len(c_fe_bins)-1, len(si_c_bins)-1))
    hist3_total = np.zeros((len(o_fe_bins)-1, len(si_fe_bins)-1))
    hist4_total = np.zeros((len(c_o_bins)-1, len(si_o_bins)-1))

    for g in range(len(gal)):
        print('fig3, adding galaxy %s' % (gal[g]))
        infile = '/home/gpruto/metal_ab/code/2dhistograms/z=%d/%s/Sodini_hist_x_HI<%.1f_n_H>%d_T<%.1f_met>%.1f.txt' % (red, gal[g], 0.1, -2, 4.3, -4)
        try:
            x_bins, y_bins, hist1, hist2, hist3, hist4 = np.loadtxt(infile, skiprows=1, usecols=(0,1,2,3,4,5), unpack=True)
            nx = int(x_bins.max()+1)
            ny = int(y_bins.max()+1)
            hist1 = hist1.reshape(nx, ny)
            hist2 = hist2.reshape(nx, ny)
            hist3 = hist3.reshape(nx, ny)
            hist4 = hist4.reshape(nx, ny)

            hist1_total += hist1
            hist2_total += hist2
            hist3_total += hist3
            hist4_total += hist4

        except OSError:
            print(f"File not found: {infile}")
            continue

    fig, ax = mlib.plot_Sodini_all()
    ax[0].imshow(hist1_total.T, origin='lower', extent=[c_fe_bins[0], c_fe_bins[-1], o_fe_bins[0], o_fe_bins[-1]], aspect='auto', cmap=cmc.batlowK, norm=matplotlib.colors.LogNorm())
    ax[1].imshow(hist2_total.T, origin='lower', extent=[c_fe_bins[0], c_fe_bins[-1], si_c_bins[0], si_c_bins[-1]], aspect='auto', cmap=cmc.batlowK, norm=matplotlib.colors.LogNorm())
    ax[2].imshow(hist3_total.T, origin='lower', extent=[o_fe_bins[0], o_fe_bins[-1], si_fe_bins[0], si_fe_bins[-1]], aspect='auto', cmap=cmc.batlowK, norm=matplotlib.colors.LogNorm())
    ax[3].imshow(hist4_total.T, origin='lower', extent=[c_o_bins[0], c_o_bins[-1], si_o_bins[0], si_o_bins[-1]], aspect='auto', cmap=cmc.batlowK, norm=matplotlib.colors.LogNorm())

    ax[0].set_xlim(c_fe_bins[0], c_fe_bins[-1])
    ax[0].set_ylim(o_fe_bins[0], o_fe_bins[-1])
    ax[1].set_xlim(c_fe_bins[0], c_fe_bins[-1])
    ax[1].set_ylim(si_c_bins[0], si_c_bins[-1])
    ax[2].set_xlim(o_fe_bins[0], o_fe_bins[-1])
    ax[2].set_ylim(si_fe_bins[0], 1.5)
    ax[3].set_xlim(c_o_bins[0], c_o_bins[-1])
    ax[3].set_ylim(si_o_bins[0], si_o_bins[-1])

    fig.savefig('/home/gpruto/metal_ab/images/paper/allgal_dlasel_hist_z%d.png' % (red), bbox_inches='tight', dpi=300)
    fig.savefig('/home/gpruto/metal_ab/images/all_gas/neutral_f>%.1f/T<%.0e_n>%.1f/met>-4/allgal_hist_z%d.png' % (0.1, 2e4, -2, red), bbox_inches='tight', dpi=300)
##########################


##### PLOT DLA SELECTION FIGURE
if fig1:
    x_hi_bins = np.linspace(0,1, 300)
    n_h_bins = np.linspace(-6, 5, 300)
    T_bins = np.linspace(1, 7.5, 300)
    met_bins = np.linspace(-15, 1, 300)

    hist1_tot_allg = np.zeros((len(n_h_bins)-1, len(x_hi_bins)-1))
    hist2_tot_allg = np.zeros((len(T_bins)-1, len(x_hi_bins)-1))
    hist3_tot_allg = np.zeros((len(met_bins)-1, len(x_hi_bins)-1))
    hist4_tot_allg = np.zeros((len(n_h_bins)-1, len(T_bins)-1))
    hist1_allg = np.zeros((len(n_h_bins)-1, len(x_hi_bins)-1))
    hist2_allg = np.zeros((len(T_bins)-1, len(x_hi_bins)-1))
    hist3_allg = np.zeros((len(met_bins)-1, len(x_hi_bins)-1))
    hist4_allg = np.zeros((len(n_h_bins)-1, len(T_bins)-1))


    for g in range(len(gal)):
        print('fig1, adding galaxy %s' % (gal[g]))
        infile = '/home/gpruto/metal_ab/code/2dhistograms/z=%d/%s/dlacuts_hist_x_HI<%.1f_n_H>%d_T<%.1f_met>%.1f.txt' % (red, gal[g], 0.1, -2, 4.3, -4)
        try:
            x_bins, y_bins, hist1_tot, hist2_tot, hist3_tot, hist4_tot, hist1, hist2, hist3, hist4 = np.loadtxt(infile, skiprows=1, usecols=(0,1,2,3,4,5,6,7,8,9), unpack=True)
            nx = int(x_bins.max()+1)
            ny = int(y_bins.max()+1)
            hist1_tot = hist1_tot.reshape(nx, ny)
            hist2_tot = hist2_tot.reshape(nx, ny)
            hist3_tot = hist3_tot.reshape(nx, ny)
            hist4_tot = hist4_tot.reshape(nx, ny)
            hist1 = hist1.reshape(nx, ny)
            hist2 = hist2.reshape(nx, ny)
            hist3 = hist3.reshape(nx, ny)
            hist4 = hist4.reshape(nx, ny)

            hist1_tot_allg += hist1_tot
            hist2_tot_allg += hist2_tot
            hist3_tot_allg += hist3_tot 
            hist4_tot_allg += hist4_tot
            hist1_allg += hist1
            hist2_allg += hist2
            hist3_allg += hist3
            hist4_allg += hist4

        except OSError:
            print(f"File not found: {infile}")
            continue

    fig_dla, ax_dla = plt.subplots(1,4, figsize=(18, 4.5))

    ax_dla[0].imshow(hist1_tot_allg.T, origin='lower', extent=[n_h_bins[0], n_h_bins[-1], x_hi_bins[0], x_hi_bins[-1]], aspect='auto', cmap=cmc.batlowK, norm=matplotlib.colors.LogNorm(), alpha=0.3)
    ax_dla[1].imshow(hist2_tot_allg.T, origin='lower', extent=[T_bins[0], T_bins[-1], x_hi_bins[0], x_hi_bins[-1]], aspect='auto', cmap=cmc.batlowK, norm=matplotlib.colors.LogNorm(), alpha=0.3)
    ax_dla[2].imshow(hist3_tot_allg.T, origin='lower', extent=[met_bins[0], met_bins[-1], x_hi_bins[0], x_hi_bins[-1]], aspect='auto', cmap=cmc.batlowK, norm=matplotlib.colors.LogNorm(), alpha=0.3)
    ax_dla[3].imshow(hist4_tot_allg.T, origin='lower', extent=[n_h_bins[0], n_h_bins[-1], T_bins[0], T_bins[-1]], aspect='auto', cmap=cmc.batlowK, norm=matplotlib.colors.LogNorm(), alpha=0.3)

    ax_dla[0].imshow(hist1_allg.T, origin='lower', extent=[n_h_bins[0], n_h_bins[-1], x_hi_bins[0], x_hi_bins[-1]], aspect='auto', cmap=cmc.batlowK, norm=matplotlib.colors.LogNorm())
    ax_dla[1].imshow(hist2_allg.T, origin='lower', extent=[T_bins[0], T_bins[-1], x_hi_bins[0], x_hi_bins[-1]], aspect='auto', cmap=cmc.batlowK, norm=matplotlib.colors.LogNorm())
    ax_dla[2].imshow(hist3_allg.T, origin='lower', extent=[met_bins[0], met_bins[-1], x_hi_bins[0], x_hi_bins[-1]], aspect='auto', cmap=cmc.batlowK, norm=matplotlib.colors.LogNorm())
    ax_dla[3].imshow(hist4_allg.T, origin='lower', extent=[n_h_bins[0], n_h_bins[-1], T_bins[0], T_bins[-1]], aspect='auto', cmap=cmc.batlowK, norm=matplotlib.colors.LogNorm())

    ax_dla[0].plot([-6, 5], [0.1, 0.1], color='k', ls='--', lw=2)
    ax_dla[0].vlines(-2, 0, 1, color='k', ls='--', lw=2)

    ax_dla[1].plot([1, 7.5], [0.1, 0.1], color='k', ls='--', lw=2)
    ax_dla[1].vlines(4.3, 0, 1, color='k', ls='--', lw=2)

    ax_dla[2].plot([-15, 1], [0.1, 0.1], color='k', ls='--', lw=2)
    ax_dla[2].vlines(-4, 0, 1, color='k', ls='--', lw=2)

    ax_dla[3].plot([-6, 5], [4.3, 4.3], color='k', ls='--', lw=2)
    ax_dla[3].plot([-2, -2], [1, 7.5], color='k', ls='--', lw=2)

    ax_dla[0].set_xlabel(r'$\log_{10} (n_{\rm H}$ [cm$^{-3}$])')
    ax_dla[1].set_xlabel(r'$\log_{10} (T$ [K])')
    ax_dla[2].set_xlabel(r'$\log_{10} (Z/Z_{\odot}$)')
    ax_dla[3].set_xlabel(r'$\log_{10} (n_{\rm H}$ [cm$^{-3}$])')

    ax_dla[0].set_ylabel(r'$n_{\rm HI}/n_{\rm H}$')
    ax_dla[1].set_ylabel(r'$n_{\rm HI}/n_{\rm H}$')
    ax_dla[2].set_ylabel(r'$n_{\rm HI}/n_{\rm H}$')
    ax_dla[3].set_ylabel(r'$\log_{10} (T$ [K])')

    ax_dla[0].set_ylim(-0.01,1.01)
    ax_dla[1].set_ylim(-0.01,1.01)
    ax_dla[2].set_ylim(-0.01,1.01)
    ax_dla[3].set_ylim(1, 7.5)

    ax_dla[0].set_xlim(-6,5)
    ax_dla[1].set_xlim(1,7.5)
    ax_dla[2].set_xlim(-15,1)
    ax_dla[3].set_xlim(-6,5)

    fig_dla.savefig('/home/gpruto/metal_ab/images/paper/allgal_dlaselection_z%d.png' % (red), bbox_inches='tight', dpi=300)
###############################################



if figpopIII:
    #### shaded plots
    c_fe_bins = np.linspace(-1.5, 1.8, 300)
    o_fe_bins = np.linspace(-1.5, 2.5, 300)
    si_c_bins = np.linspace(-1.5, 1., 300)
    si_fe_bins = np.linspace(-0.5, 1.2, 300)
    si_o_bins = np.linspace(-1.5, 1.2, 300)
    c_o_bins = np.linspace(-1.2, 1.2, 300)

    hist1_total = np.zeros((len(c_fe_bins)-1, len(o_fe_bins)-1))
    hist2_total = np.zeros((len(c_fe_bins)-1, len(si_c_bins)-1))
    hist3_total = np.zeros((len(o_fe_bins)-1, len(si_fe_bins)-1))
    hist4_total = np.zeros((len(c_o_bins)-1, len(si_o_bins)-1))

    for g in range(len(gal)):
        infile = '/home/gpruto/metal_ab/code/2dhistograms/z=%d/%s/Sodini_hist_x_HI<%.1f_n_H>%d_T<%.1f_met>%.1f.txt' % (red, gal[g], 0.1, -2, 4.3, -4)
        try:
            x_bins, y_bins, hist1, hist2, hist3, hist4 = np.loadtxt(infile, skiprows=1, usecols=(0,1,2,3,4,5), unpack=True)
            nx = int(x_bins.max()+1)
            ny = int(y_bins.max()+1)
            hist1 = hist1.reshape(nx, ny)
            hist2 = hist2.reshape(nx, ny)
            hist3 = hist3.reshape(nx, ny)
            hist4 = hist4.reshape(nx, ny)

            hist1_total += hist1
            hist2_total += hist2
            hist3_total += hist3
            hist4_total += hist4

        except OSError:
            print(f"File not found: {infile}")
            continue

    fig, ax = mlib.plot_Sodini_all(special_only=True, WW_model=False) #represent only points that are possible popIII
    ax[0].imshow(hist1_total.T, origin='lower', extent=[c_fe_bins[0], c_fe_bins[-1], o_fe_bins[0], o_fe_bins[-1]], aspect='auto', cmap=cmc.batlowK, norm=matplotlib.colors.LogNorm(), alpha=0.3)
    ax[1].imshow(hist2_total.T, origin='lower', extent=[c_fe_bins[0], c_fe_bins[-1], si_c_bins[0], si_c_bins[-1]], aspect='auto', cmap=cmc.batlowK, norm=matplotlib.colors.LogNorm(), alpha=0.3)
    ax[2].imshow(hist3_total.T, origin='lower', extent=[o_fe_bins[0], o_fe_bins[-1], si_fe_bins[0], si_fe_bins[-1]], aspect='auto', cmap=cmc.batlowK, norm=matplotlib.colors.LogNorm(), alpha=0.3)
    ax[3].imshow(hist4_total.T, origin='lower', extent=[c_o_bins[0], c_o_bins[-1], si_o_bins[0], si_o_bins[-1]], aspect='auto', cmap=cmc.batlowK, norm=matplotlib.colors.LogNorm(), alpha=0.3)

    ax[0].set_xlim(c_fe_bins[0], c_fe_bins[-1])
    ax[0].set_ylim(o_fe_bins[0], o_fe_bins[-1])
    ax[1].set_xlim(c_fe_bins[0], c_fe_bins[-1])
    ax[1].set_ylim(si_c_bins[0], si_c_bins[-1])
    ax[2].set_xlim(o_fe_bins[0], o_fe_bins[-1])
    ax[2].set_ylim(si_fe_bins[0], 1.5)
    ax[3].set_xlim(c_o_bins[0], c_o_bins[-1])
    ax[3].set_ylim(si_o_bins[0], si_o_bins[-1])

    #box with possible popIII
    popIII_left = 0.24
    popIII_right = 0.81
    popIII_down = 0.36
    popIII_up = 0.86
    ax[3].plot([popIII_left, popIII_right], [popIII_down, popIII_down], color='black', linestyle='--')
    ax[3].plot([popIII_left, popIII_right], [popIII_up, popIII_up], color='black', linestyle='--')
    ax[3].vlines(popIII_left, popIII_down, popIII_up, color='black', linestyle='--')
    ax[3].vlines(popIII_right, popIII_down, popIII_up, color='black', linestyle='--')

    for g in range(len(gal)):
        print('figpopIII, adding galaxy %s' % (gal[g]))
        infile = '/home/gpruto/metal_ab/code/regions/popIII/z=%d/%s/%s_z%.1f_regionpopIII.txt' % (red, gal[g], gal[g], red)
        with open(infile, 'r') as f:
            header = f.readline()  # Read the header line
            data = np.loadtxt(infile, skiprows=1, usecols=(0,1,2,3,4,5,6,7,8,9,10), unpack=True)

            if len(data[0]) == 0:
                print(f"No data found in {infile}. Skipping this file.")
                continue

            data = data.T
            coords = data[:, 0:3]
            carbon_density = data[:, 7]
            oxygen_density = data[:, 8]
            silicon_density = data[:, 9]
            iron_density = data[:, 10]
            c_fe_popIII = np.log10(carbon_density/iron_density) - mlib.C_Fe_solar
            o_fe_popIII = np.log10(oxygen_density/iron_density) - mlib.O_Fe_solar
            si_c_popIII = np.log10(silicon_density/carbon_density) - mlib.Si_C_solar
            si_fe_popIII = np.log10(silicon_density/iron_density) - mlib.Si_Fe_solar
            c_o_popIII = np.log10(carbon_density/oxygen_density) - mlib.C_O_solar
            si_o_popIII = np.log10(silicon_density/oxygen_density) - mlib.Si_O_solar

            ax[0].scatter(c_fe_popIII, o_fe_popIII, color='royalblue', s=1)
            ax[1].scatter(c_fe_popIII, si_c_popIII, color='royalblue', s=1)
            ax[2].scatter(o_fe_popIII, si_fe_popIII, color='royalblue', s=1)
            ax[3].scatter(c_o_popIII, si_o_popIII, color='royalblue', s=1)

    fig.savefig('/home/gpruto/metal_ab/images/paper/allgal_popIII_hist_z%d.png' % (red), bbox_inches='tight', dpi=300)

##########################

