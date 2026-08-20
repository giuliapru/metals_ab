import numpy as np
import matplotlib.pyplot as plt
from sympy import im
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
import pandas as pd

gal = ['g5229300', 'g2274036', 'g519761', 'g500531', 'g137030', 'g37591','g33206', 'g10304', 'g5760', 'g1163', 'g578', 'g205', 'g39', 'g2']
gal_names = ['m8.2', 'm8.5', 'm8.9', 'm9.3', 'm9.7', 'm10.0', 'm10.4', 'm10.8', 'm11.1', 'm11.5', 'm11.9', 'm12.2', 'm12.6', 'm13.0']
reds = [8, 7, 6, 5]
colors = cmc.batlowK(np.linspace(0, 1, len(gal)))
gal_cmap = matplotlib.colors.ListedColormap(colors)
gal_bounds = np.arange(len(gal) + 1) - 0.5
gal_norm = matplotlib.colors.BoundaryNorm(gal_bounds, gal_cmap.N)


def draw_medians(y, ax):
    for z in range(len(reds)):
        point_z = []
        for g in range(len(gal)):
            point_z.append(y[g,z])
        
        ymed = (np.nanmedian(point_z))
        yerr_low = ymed - (np.nanpercentile(point_z, 16))
        yerr_high = (np.nanpercentile(point_z, 84)) - ymed
        ax.errorbar(reds[z], ymed, yerr=[[yerr_low], [yerr_high]], fmt='none', ecolor='black', zorder=1)
        ax.scatter(reds[z], ymed, c='gray', edgecolors='black', s=20**2, marker='*', alpha=0.7, zorder=2, label='median' if z == 0 else None)
        ax.legend()

#### PER PTC NUMBER
infile = '/home/gpruto/metal_ab/code/DLA_cut/dla_cuts_x_HI>%.1f_n_H>%d_T<%.1f_met>%.1f.txt' %(0.1, -2, 4.3, -4)

data = pd.read_csv(
    infile,
    delim_whitespace=True,
    header=None,
    comment='#',
    names=['galaxy', 'redshift', 'fraction_of_DLA_particles', 'DLA_in_rvir', 'DLA_in_1p5rvir'],
)
data['redshift'] = data['redshift'].astype(int)

table = data.pivot(index='galaxy', columns='redshift', values='fraction_of_DLA_particles')
dla_frac = table.reindex(index=gal, columns=reds).to_numpy()
table_dla_in_rvir = data.pivot(index='galaxy', columns='redshift', values='DLA_in_rvir')
dla_in_rvir = table_dla_in_rvir.reindex(index=gal, columns=reds).to_numpy()

###Fraction of DLA particles
fig, ax = plt.subplots(1, 1, figsize=(5,5))

for z in range(len(reds)):
    for g in range(len(gal)):
        ax.scatter(reds[z]-0.3+0.3/6.5*g, np.log10(dla_frac[g,z]), c=colors[g], s=10)
draw_medians(np.log10(dla_frac), ax)

#ax.set_ylim(-0.01, 1.01)
ax.set_ylim(-6, 0)
ax.set_xticks(reds)
ax.set_xticklabels(reds)
ax.set_xlabel('Redshift')
ax.set_ylabel('Number fraction of DLA-like gas cells [log10]')
#add colorbar with galaxy names
sm = plt.cm.ScalarMappable(cmap=gal_cmap, norm=gal_norm)
cax  = fig.add_axes([0.92, 0.15, 0.04, 0.7])  # [left, bottom, width, height]
cbar = plt.colorbar(sm, cax=cax, ticks=np.arange(len(gal)))
cbar.ax.set_yticklabels(gal_names)
cbar.ax.minorticks_off()
cbar.ax.tick_params(which='minor', left=False, right=False)

fig.savefig('/home/gpruto/metal_ab/images/paper/DLA_fraction.png', dpi=300, bbox_inches='tight')
###################


#### DLA in rvir
fig_r, ax_r = plt.subplots(1, 1, figsize=(5,5))

for z in range(len(reds)):
    for g in range(len(gal)):
        ax_r.scatter(reds[z]-0.3+0.3/6.5*g, (dla_in_rvir[g,z]), c=colors[g], s=10)

draw_medians((dla_in_rvir), ax_r)
ax_r.set_ylim(-0.01, 1.01)
ax_r.set_xticks(reds)
ax_r.set_xticklabels(reds)
ax_r.set_xlabel('Redshift')
ax_r.set_ylabel(r'Number fraction of DLA-like gas cells in $R_{\rm vir}$')
#add colorbar with galaxy names
sm = plt.cm.ScalarMappable(cmap=gal_cmap, norm=gal_norm)
cax  = fig_r.add_axes([0.92, 0.15, 0.04, 0.7])  # [left, bottom, width, height]
cbar = plt.colorbar(sm, cax=cax, ticks=np.arange(len(gal)))
cbar.ax.set_yticklabels(gal_names)
cbar.ax.minorticks_off()
cbar.ax.tick_params(which='minor', left=False, right=False)
#ax_r.set_ylim(-10, 0)

fig_r.savefig('/home/gpruto/metal_ab/images/paper/DLA_in_rvir.png', dpi=300, bbox_inches='tight')
###################



#### PER PTC VOLUME
infile = '/home/gpruto/metal_ab/code/DLA_cut/dla_cuts_volume_x_HI>%.1f_n_H>%d_T<%.1f_met>%.1f.txt' %(0.1, -2, 4.3, -4)

data = pd.read_csv(
    infile,
    delim_whitespace=True,
    header=None,
    comment='#',
    names=['galaxy', 'redshift', 'fraction_of_DLA_particles', 'DLA_in_rvir', 'DLA_in_1p5rvir'],
)
data['redshift'] = data['redshift'].astype(int)

table = data.pivot(index='galaxy', columns='redshift', values='fraction_of_DLA_particles')
dla_frac = table.reindex(index=gal, columns=reds).to_numpy()
table_dla_in_rvir = data.pivot(index='galaxy', columns='redshift', values='DLA_in_rvir')
dla_in_rvir = table_dla_in_rvir.reindex(index=gal, columns=reds).to_numpy()

###Fraction of DLA particles
fig, ax = plt.subplots(1, 1, figsize=(5,5))

for z in range(len(reds)):
    for g in range(len(gal)):
        ax.scatter(reds[z]-0.3+0.3/6.5*g, np.log10(dla_frac[g,z]), c=colors[g], s=10)

draw_medians(np.log10(dla_frac), ax)
ax.set_ylim(-6,0)
ax.set_xticks(reds)
ax.set_xticklabels(reds)
ax.set_xlabel('Redshift')
ax.set_ylabel('Volume fraction of DLA-like gas cells [log10]')
#add colorbar with galaxy names
sm = plt.cm.ScalarMappable(cmap=gal_cmap, norm=gal_norm)
cax  = fig.add_axes([0.92, 0.15, 0.04, 0.7])  # [left, bottom, width, height]
cbar = plt.colorbar(sm, cax=cax, ticks=np.arange(len(gal)))
cbar.ax.set_yticklabels(gal_names)
cbar.ax.minorticks_off()
cbar.ax.tick_params(which='minor', left=False, right=False)

fig.savefig('/home/gpruto/metal_ab/images/paper/DLA_fraction_volume.png', dpi=300, bbox_inches='tight')
###################


#### DLA in rvir
fig_r, ax_r = plt.subplots(1, 1, figsize=(5,5))

for z in range(len(reds)):
    for g in range(len(gal)):
        ax_r.scatter(reds[z]-0.3+0.3/6.5*g, (dla_in_rvir[g,z]), c=colors[g], s=10)

draw_medians(dla_in_rvir, ax_r)
ax_r.set_ylim(-0.01, 1.01)
ax_r.set_xticks(reds)
ax_r.set_xticklabels(reds)
ax_r.set_xlabel('Redshift')
ax_r.set_ylabel(r'Volume fraction of DLA-like gas cells in $R_{\rm vir}$')
#add colorbar with galaxy names
sm = plt.cm.ScalarMappable(cmap=gal_cmap, norm=gal_norm)
cax  = fig_r.add_axes([0.92, 0.15, 0.04, 0.7])  # [left, bottom, width, height]
cbar = plt.colorbar(sm, cax=cax, ticks=np.arange(len(gal)))
cbar.ax.set_yticklabels(gal_names)
cbar.ax.minorticks_off()
cbar.ax.tick_params(which='minor', left=False, right=False)
#ax_r.set_ylim(-10, 0)
fig_r.savefig('/home/gpruto/metal_ab/images/paper/DLA_in_rvir_volume.png', dpi=300, bbox_inches='tight')
###################
