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
import cmcrameri.cm as cmc
import matplotlib.colors


hist_dust_to_gas = True
bigplot = False

#redshift = [3, 4, 5, 6, 7, 8, 9]
redshift = [9,8,7,6,5,4,3]
gals = ['g2', 'g39', 'g205', 'g578', 'g1163', 'g5760', 'g10304', 'g33206', 'g37591', 'g137030', 'g500531', 'g519761', 'g2274036', 'g5229300']
gals = gals[::-1] #reverse the order of the galaxies to match the order in the DLA_cut_slurmmaker.py file

def plot_hist_with_cut(x, y, dust_to_gas, dtg_cut, redshift_dtg, axs, x_bins, y_bins, gals_sel):
    ##### axs 0 and 3 - DUST TO GAS
    r = np.where(np.array(redshift) == redshift_dtg)[0][0]
    x_highdtg = []
    y_highdtg = []
    x_lowdtg = []
    y_lowdtg = []

    for g in range(len(gals)):
        high_dtg = np.log10(dust_to_gas[r][g]) > dtg_cut
        low_dtg = np.log10(dust_to_gas[r][g]) < dtg_cut

    x_highdtg.extend(np.array(x[r][g])[high_dtg])
    y_highdtg.extend(np.array(y[r][g])[high_dtg])
    x_lowdtg.extend(np.array(x[r][g])[low_dtg])
    y_lowdtg.extend(np.array(y[r][g])[low_dtg])

    x_highdtg = np.array(x_highdtg)
    y_highdtg = np.array(y_highdtg)
    x_lowdtg = np.array(x_lowdtg)
    y_lowdtg = np.array(y_lowdtg)

    mlib.hist_2d(x_highdtg, y_highdtg, axs[0], x_bins=x_bins, y_bins=y_bins, color=cmc.batlowK, alpha=1.0, output=False)
    mlib.hist_2d(x_lowdtg, y_lowdtg, axs[3], x_bins=x_bins, y_bins=y_bins, color=cmc.batlowK, alpha=1.0, output=False)
    axs[0].text(0.05, 0.92, r'log($m_{\rm dust}$/$m_{\rm gas}$) > %.1f' % dtg_cut, transform=axs[0].transAxes, fontsize=14, fontweight='bold')
    axs[3].text(0.05, 0.92, r'log($m_{\rm dust}$/$m_{\rm gas}$) < %.1f' % dtg_cut, transform=axs[3].transAxes, fontsize=14, fontweight='bold')
    axs[0].text(0.80, 0.92, 'z = %d' %redshift[r], transform=axs[0].transAxes, fontsize=14, fontweight='bold', color='tab:green')
    axs[3].text(0.80, 0.92, 'z = %d' %redshift[r], transform=axs[3].transAxes, fontsize=14, fontweight='bold', color='tab:green')


    ###### for axs 1 and 4
    x_allgal = [[] for z in range(len(redshift))]
    y_allgal = [[] for z in range(len(redshift))]

    for z in range(len(redshift)):
        for g in range(len(gals)):
            x_allgal[z].extend(x[z][g])
            y_allgal[z].extend(y[z][g])

        x_allgal[z] = np.array(x_allgal[z])
        y_allgal[z] = np.array(y_allgal[z])

    mlib.hist_2d(x_allgal[0], y_allgal[0], axs[1], x_bins=x_bins, y_bins=y_bins, color=cmc.batlowK, alpha=1.0, output=False)
    mlib.hist_2d(x_allgal[2], y_allgal[2], axs[4], x_bins=x_bins, y_bins=y_bins, color=cmc.batlowK, alpha=1.0, output=False)
    axs[1].text(0.05, 0.92, r'z = %d' %redshift[0], transform=axs[1].transAxes, fontsize=14, fontweight='bold')
    axs[4].text(0.05, 0.92, r'z = %d' %redshift[2], transform=axs[4].transAxes, fontsize=14, fontweight='bold')
    axs[1].text(0.65, 0.92, 'all regions' , transform=axs[1].transAxes, fontsize=14, fontweight='bold', color='tab:green')
    axs[4].text(0.65, 0.92, 'all regions' , transform=axs[4].transAxes, fontsize=14, fontweight='bold', color='tab:green')


    ###### for axs 2 and 5 
    mlib.hist_2d(np.array(x[r][gals_sel[0]]), np.array(y[r][gals_sel[0]]), axs[2], x_bins=x_bins, y_bins=y_bins, color=cmc.batlowK, alpha=1.0, output=False)
    mlib.hist_2d(np.array(x[r][gals_sel[1]]), np.array(y[r][gals_sel[1]]), axs[5], x_bins=x_bins, y_bins=y_bins, color=cmc.batlowK, alpha=1.0, output=False)
    axs[2].text(0.05, 0.92, r'%s' %gals[gals_sel[0]], transform=axs[2].transAxes, fontsize=14, fontweight='bold')
    axs[5].text(0.05, 0.92, r'%s' %gals[gals_sel[1]], transform=axs[5].transAxes, fontsize=14, fontweight='bold')
    axs[2].text(0.80, 0.92, 'z = %d' %redshift[1], transform=axs[2].transAxes, fontsize=14, fontweight='bold', color='tab:green')
    axs[5].text(0.80, 0.92, 'z = %d' %redshift[1], transform=axs[5].transAxes, fontsize=14, fontweight='bold', color='tab:green')





if hist_dust_to_gas:
    for z in range(len(redshift)):
        fig, ax = plt.subplots(3, 5, figsize=(25, 15))
        ax = ax.flatten()
        for g in range(len(gals)):
            infile = '/home/gpruto/metal_ab/code/DLA_cut/z=%d/%s/dla_cuts_x_HI>0.1_n_H>-2_T<4.3_met>-4.0.txt' %(redshift[z], gals[g])
            if os.path.exists(infile):
                data = np.loadtxt(infile, unpack=True, skiprows=1, max_rows=10)
                data = data.T
                print(redshift[z], gals[g])
                if len(data) == 0:
                    print('No data for %s at z=%d' %(gals[g], redshift[z]))
                    continue
                if len(data[0])==14:
                    print('All good for %s at z=%d' %(gals[g], redshift[z]))
                elif len(data[0])<14:
                    print('We do not have all the data for %s at z=%d' %(gals[g], redshift[z]))
                    continue
                '''dust_to_gas = data[:, 13]

                ax[g].hist(np.log10(dust_to_gas), bins=50, density=True, histtype='step', color='black')
                ax[g].set_xlabel(r'$\log_{10}(M_{dust}/M_{gas})$')
                if g == 0 or g == 5 or g == 10:
                    ax[g].set_ylabel('Probability Density')
                ax[g].set_title('%s' % gals[g])'''

        #don't show the last subplot
        ax[-1].set_visible(False)
        fig.savefig('/home/gpruto/metal_ab/images/dust_effects/dust_to_gas_hist_z=%d.png' % redshift[z], dpi=300, bbox_inches='tight')


if bigplot:
    dtg_cut = -4.5
    redshift = [4,6,8]

    dust_to_gas = [[[] for g in range(len(gals))] for z in range(len(redshift))]
    si_o = [[[] for g in range(len(gals))] for z in range(len(redshift))]
    c_o = [[[] for g in range(len(gals))] for z in range(len(redshift))]
    c_fe = [[[] for g in range(len(gals))] for z in range(len(redshift))]
    o_fe = [[[] for g in range(len(gals))] for z in range(len(redshift))]
    si_fe = [[[] for g in range(len(gals))] for z in range(len(redshift))]
    si_c = [[[] for g in range(len(gals))] for z in range(len(redshift))]

    for z in range(len(redshift)):
        for g in range(len(gals)):
            infile = '/home/gpruto/metal_ab/code/DLA_cut/z=%d/%s/dla_cuts_x_HI>0.1_n_H>-2_T<4.3_met>-4.0.txt' %(redshift[z], gals[g])
            if os.path.exists(infile):
                data = np.loadtxt(infile, unpack=True, skiprows=1)
                data = data.T
                if len(data) == 0:
                    print('No data for %s at z=%d' %(gals[g], redshift[z]))
                    continue
                if len(data[0])==13:
                    print('All good for %s at z=%d' %(gals[g], redshift[z]))
                elif len(data[0])<13:
                    print('We do not have all the data for %s at z=%d' %(gals[g], redshift[z]))
                    continue
                dust_to_gas[z][g] = data[:, 13]
                c_o[z][g] = data[:,7]
                si_o[z][g] = data[:, 8]
                c_fe[z][g] = data[:, 9]
                o_fe[z][g] = data[:, 10]
                si_fe[z][g] = data[:, 11]
                si_c[z][g] = data[:, 12]
            else:
                print('We do not have data for this galaxy at this redshift')

            
    c_fe_bins = np.linspace(-1.5, 1.8, 300)
    o_fe_bins = np.linspace(-1.5, 2.5, 300)
    si_c_bins = np.linspace(-1.5, 1., 300)
    si_fe_bins = np.linspace(-0.5, 1.2, 300)
    si_o_bins = np.linspace(-1.5, 1.2, 300)
    c_o_bins = np.linspace(-1.2, 1.2, 300)

    x_bins = [c_fe_bins, c_fe_bins, o_fe_bins, c_o_bins]
    y_bins = [o_fe_bins, si_c_bins, si_fe_bins, si_o_bins]
    ab_plot_x = [c_fe, c_fe, o_fe, c_o]
    ab_plot_y = [o_fe, si_c, si_fe, si_o]
    ab_plot_xlabel = ['[C/Fe]', '[C/Fe]', '[O/Fe]', '[C/O]']
    ab_plot_ylabel = ['[O/Fe]', '[Si/C]', '[Si/Fe]', '[Si/O]']
    ab_plot_title = ['a', 'b', 'c', 'd']

    for ab in range(len(ab_plot_x)):
        figs, axs = plt.subplots(2, 3, figsize=(15, 10))
        figs.subplots_adjust(wspace=0, hspace = 0)
        axs = axs.flatten()

        for i in range(len(axs)):
            axs[i].vlines(0, np.min(y_bins[ab]), np.max(y_bins[ab]), ls='--', color='gray')
            axs[i].plot([np.min(x_bins[ab]), np.max(x_bins[ab])], [0,0], ls='--', color='gray')
            axs[i].tick_params(direction = "in", axis="both", left=True, right=True, bottom=True, top=True)
            if i%3!=0:
                axs[i].set_yticklabels([])

            axs[i].set_xlim(np.min(x_bins[ab]), np.max(x_bins[ab]))
            axs[i].set_ylim(np.min(y_bins[ab]), np.max(y_bins[ab]))
            if i>=3:
                axs[i].set_xlabel(ab_plot_xlabel[ab])
            if i%3 == 0:
                axs[i].set_ylabel(ab_plot_ylabel[ab])

        plot_hist_with_cut(ab_plot_x[ab], ab_plot_y[ab], dust_to_gas, dtg_cut, 6, axs, x_bins[ab], y_bins[ab], gals_sel=[2,9])
        figs.savefig('/home/gpruto/metal_ab/images/dust_effects/dtgcut_%s_dtgcut=%.1f.png' %(ab_plot_title[ab], dtg_cut), dpi=300, bbox_inches='tight')

