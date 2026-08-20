'''This piece of code is to compare results from our simulations to the high-redshift absorption systems 
in Pollock2026, Nakane 2026, Zhu 2026'''
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('/home/gpruto/CGM_galaxies/paper.style')
import cmcrameri.cm as cmc
import sys

red = int(sys.argv[1])

#colors_gal = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
#5 colors
colors_gal =[ "#0656A0", "#21A640", '#F6AE2D', "#D65108", '#591F0A']
gal_obs = ['Gz9p3', 'CEERS-1019', 'GLASS-100003', 'GLASS-10021', 'EGS-z7p2']


##Pollock 2026
Pollock_sio = np.array([-0.26, 0.08, 0.28])
Pollock_co = np.array([0.19, 1.07, 0.69])
Pollock_sio_error = np.array([0.29, 0.16, 0.55])
Pollock_co_error = np.array([0.3, 0.16, 0.56])
colors_Pollock = colors_gal[:len(Pollock_sio)]

##nakane 2026
Nakane_sio = np.array([0.62, 0.89, 0.43, 0.3, 0.69])
Nakane_co = np.array([0.35, 1.11, 0.58, -0.35, 0.56])
Nakane_sio_up = np.array([0.21, 0.52, 0.72, 0.35, 0.69])
Nakane_sio_low = np.array([0.21, 0.43, 1.63, 0.76, 0.35])
Nakane_co_up = np.array([0.75, 1.48, 1.32, 0.3, 0.93])
Nakane_co_low = np.array([0.38, 0.37, 0.94, 0.9, 0.47])
colors_Nakane = colors_gal

###zhu 2026
Zhu_sio = np.array([-0.16, 0.13, -0.47])
Zhu_co = np.array([0.1, 0, -0.2])
Zhu_sio_err = np.array([0.1, 0.14, 0.09])
Zhu_co_err = np.array([0.08, 0.15, 0.056])
colors_Zhu = [colors_gal[i] for i in [0, 2, 3]]




fig, ax = plt.subplots(1,3, figsize = (15, 5))
titles = ['Zhu 2026', 'Pollock 2026', 'Nakane 2026']

for aa in ax:
    aa.set_xlabel('[C/O]')
    aa.set_ylabel('[Si/O]')
    aa.set_xlim(-1, 2)
    aa.set_ylim(-1, 2)
    aa.axhline(0, color='grey', ls='--')
    aa.axvline(0, color='grey', ls='--')
    aa.text(0.05, 0.95, titles[np.where(ax==aa)[0][0]], transform=aa.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

for j in range(len(Nakane_sio)):
    if j < len(Zhu_sio):
        ax[0].errorbar([Zhu_co[j]], [Zhu_sio[j]], xerr=[Zhu_co_err[j]], yerr=[Zhu_sio_err[j]], fmt='o', ms=7, color=colors_Zhu[j])
        ax[1].errorbar([Pollock_co[j]], [Pollock_sio[j]], xerr=[Pollock_co_error[j]], yerr=[Pollock_sio_error[j]], fmt='o', ms=7, color=colors_Pollock[j])
    ax[2].errorbar([Nakane_co[j]], [Nakane_sio[j]], xerr=[[Nakane_co_low[j]], [Nakane_co_up[j]]], yerr=[[Nakane_sio_low[j]], [Nakane_sio_up[j]]], fmt='o', ms=7, color=colors_Nakane[j], label=gal_obs[j])


gal = ['g5229300', 'g2274036', 'g519761', 'g500531', 'g137030', 'g37591','g33206', 'g10304', 'g5760', 'g1163', 'g578', 'g205', 'g39', 'g2']
si_o_bins = np.linspace(-1.5, 2, 400)
c_o_bins = np.linspace(-1.3, 2, 300)
hist_4_tot = np.zeros((len(c_o_bins)-1, len(si_o_bins)-1))

for g in range(len(gal)):
    infile = '/home/gpruto/metal_ab/code/2dhistograms/z=%d/%s/comphighz_x_HI<%.1f_n_H>%d_T<%.1f_met>%.1f.txt' % (red, gal[g], 0.1, -2, 4.3, -4)
    try:
        x_bins, y_bins, hist4 = np.loadtxt(infile, skiprows=1, unpack=True)
        nx = int(x_bins.max()+1)
        ny = int(y_bins.max()+1)
        hist4 = hist4.reshape(nx, ny)

        hist_4_tot += hist4

    except OSError:
        print(f"File not found: {infile}")
        continue

for i in range(len(ax)):
    ax[i].imshow(hist_4_tot.T, origin='lower', extent=[c_o_bins[0], c_o_bins[-1], si_o_bins[0], si_o_bins[-1]], aspect='auto', cmap=cmc.batlowK, norm=matplotlib.colors.LogNorm(), alpha = 0.6)

ax[2].legend()

plt.savefig('/home/gpruto/metal_ab/images/comparison_highzabs/comparison_highzabs_z=%d.png' %red, dpi=300, bbox_inches='tight')
plt.savefig('/home/gpruto/metal_ab/images/paper/comparison_highzabs_z=%d.png' %red, dpi=300, bbox_inches='tight')
