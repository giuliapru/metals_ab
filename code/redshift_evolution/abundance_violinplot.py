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

snaps = [188, 129, 92, 68, 51, 39, 30]
reds = [3, 4,5,6,7,8, 9]
colors = cmc.batlowK(np.linspace(0, 1, 14))
gals = ['g5229300', 'g2274036', 'g519761', 'g500531', 'g137030', 'g37591','g33206', 'g10304', 'g5760', 'g1163', 'g578', 'g205', 'g39', 'g2']
run = 'fiducial' 
cond_hr = 0.5
gal = str(sys.argv[1])
color_violin = colors[np.where(np.array(gals) == gal)[0][0]]

#### DLA-like gas selection criteria
neutral_frac_min = 0.1
n_H_min = -2
T_max  = 2e4
metallicity_min = -4
Z_solar = 0.0127

#### WEIGHTS
weights = True #if True use gas cell volume as weights


test_for_allgal = False #if True, run for g33206 and g10304 together just to test weather the code allgal is okay
if test_for_allgal:
    gals = ['g33206', 'g10304']
    color_violin = 'tab:orange'

def clean(arr):
    arr = arr[np.isfinite(arr)]
    return arr

def abundances_for_snap(targethalo, snap):
    ### READ DATA
    targethalo.read_haloes(snap, 0)
    _, volume, redshift, _, _, _, h_density, hi_density, carbon_density, oxygen_density, silicon_density, iron_density, temperature, metallicity = targethalo.gas_properties(snap, cond_hr, 1., all=True, metals=True, temperature=True, metallicity=True)
    dla_cond = ((hi_density/h_density) > neutral_frac_min) & (np.log10(h_density) > n_H_min) & (temperature < T_max) & (np.log10(metallicity/Z_solar) > metallicity_min)

    # compute ratios on masked data; silence invalid/inf warnings and then drop non-finite
    with np.errstate(divide='ignore', invalid='ignore'):
        si_o = (np.log10(silicon_density[dla_cond] / oxygen_density[dla_cond]) - mlib.Si_O_solar)
        c_o  = (np.log10(carbon_density[dla_cond]  / oxygen_density[dla_cond]) - mlib.C_O_solar)
        c_fe = (np.log10(carbon_density[dla_cond]  / iron_density[dla_cond])   - mlib.C_Fe_solar)
        o_fe = (np.log10(oxygen_density[dla_cond]  / iron_density[dla_cond])   - mlib.O_Fe_solar)
        si_fe= (np.log10(silicon_density[dla_cond] / iron_density[dla_cond])   - mlib.Si_Fe_solar)
        c_si = -((np.log10(silicon_density[dla_cond] / carbon_density[dla_cond]) - mlib.Si_C_solar))

    volume = volume[dla_cond]
    
    return si_o, c_o, c_fe, o_fe, si_fe, c_si, redshift, volume


def weighted_kde(values, weights, y, bw):
    values = np.asarray(values)
    weights = np.asarray(weights)
    
    mask = np.isfinite(values) & np.isfinite(weights)
    values = values[mask]
    weights = weights[mask]

    W = np.sum(weights)
    weights = weights / W

    kde = gaussian_kde(values, weights=weights, bw_method=bw)

    #find 16th, 84th percentiles and median of the weigthed distribution
    median = np.percentile(values, 50, weights=weights, method='inverted_cdf')
    p16 = np.percentile(values, 16, weights=weights, method='inverted_cdf')
    p84 = np.percentile(values, 84, weights=weights, method='inverted_cdf')
    
    return kde(y), W, p16, p84, median


def plot_violin(ax, y, density, p16, p84, median, x=0, color='C0', alpha=0.6):

    #normalise such that the maximum width is 0.4
    density = density / np.max(density) * 0.36
    ax.fill_betweenx(y, x - density, x + density, color=color, alpha=alpha)

    # optional percentile lines
    ax.vlines(x, p16, p84, color=color, lw=1.5)

    #draw horizontal median line and 16th and 84th percentile lines
    ax.hlines(median, x - 0.3, x + 0.3, color=color, lw=1.5)
    ax.hlines(p16, x - 0.3, x + 0.3, color=color, lw=1.5)
    ax.hlines(p84, x - 0.3, x + 0.3, color=color, lw=1.5)



targethalo = TargetHalo(gal, run)

si_o_allz = []
c_o_allz = []
c_fe_allz = []
o_fe_allz = []
si_fe_allz = []
c_si_allz = []
volume_allz = []

for snap in snaps:
    si_o, c_o, c_fe, o_fe, si_fe, c_si, redshift, volume = abundances_for_snap(targethalo, snap)
    si_o_allz.append(si_o)
    c_o_allz.append(c_o)
    c_fe_allz.append(c_fe)
    o_fe_allz.append(o_fe)
    si_fe_allz.append(si_fe)
    c_si_allz.append(c_si)
    volume_allz.append(volume)

fig, ax = plt.subplots(3,2, figsize = (12, 15))
ax = ax.flatten()

labels = ['[C/Si]', '[C/O]', '[C/Fe]', '[O/Fe]', '[Si/Fe]', '[Si/O]']
data = [c_si_allz, c_o_allz, c_fe_allz, o_fe_allz, si_fe_allz, si_o_allz]
intervals = [np.linspace(-0.4, 0.5, 500), np.linspace(-1, 0.2, 500),
                np.linspace(-0.3, 1, 500), np.linspace(0, 1.7, 500),
                np.linspace(-0.2, 0.8, 500), np.linspace(-1.2, 0.4, 500)]


redshift_intervals = [2, 4.5, 6.5]
grey_region_mean = [mlib.c_si_sod_mean, mlib.c_o_sod_mean, mlib.c_fe_sod_mean, mlib.o_fe_sod_mean, mlib.si_fe_sod_mean, mlib.si_o_sod_mean]
grey_region_sigma_mean = [mlib.c_si_sod_mean_err, mlib.c_o_sod_mean_err, mlib.c_fe_sod_mean_err, mlib.o_fe_sod_mean_err, mlib.si_fe_sod_mean_err, mlib.si_o_sod_mean_err]
grey_region_sigma = [mlib.c_si_sod_sigma, mlib.c_o_sod_sigma, mlib.c_fe_sod_sigma, mlib.o_fe_sod_sigma, mlib.si_fe_sod_sigma, mlib.si_o_sod_sigma]


for i in range(len(ax)):
    for j in range(2):
        ax[i].fill_betweenx((grey_region_mean[i][j]-grey_region_sigma[i][j], grey_region_mean[i][j]+grey_region_sigma[i][j]), x1=redshift_intervals[j], x2=redshift_intervals[j+1], color='grey', alpha=0.3)
        ax[i].fill_betweenx((grey_region_mean[i][j]-grey_region_sigma_mean[i][j], grey_region_mean[i][j]+grey_region_sigma_mean[i][j]), x1=redshift_intervals[j], x2=redshift_intervals[j+1], color='grey', alpha=0.8)

#without weights
if weights==False:
    for i in range(len(ax)):

        mask = np.ones(len(data[i]), dtype=bool)
        for j in range(len(data[i])):
            data[i][j] = clean(data[i][j])
            mask[j] = len(data[i][j]) > 0

        selected = [data_i for data_i, m in zip(data[i], mask) if m]
        reds_sel = np.array(reds)[mask]
        print(reds_sel)
        
        q = [(0.16, 0.84)] * len(selected)
        vp = ax[i].violinplot(selected, positions=reds_sel, showmeans=False, showmedians=True, showextrema=False, quantiles = q)
        for body in vp['bodies']:
            body.set_facecolor(color_violin)
            body.set_edgecolor(color_violin)
        
        vp["cquantiles"].set_color(color_violin)
        vp["cquantiles"].set_linewidth(1.5)
        vp["cmedians"].set_color(color_violin)
        vp["cmedians"].set_linewidth(1.5)

        for x0, vals in zip(reds_sel, selected):
            p16, p84 = np.percentile(vals, [16, 84])
            ax[i].vlines(x0, p16, p84, color=color_violin, lw=1.5, alpha=0.8)

        xticks = [2,3,4,5,6,7,8,9]
        ax[i].set_xlabel('Redshift')
        ax[i].set_ylabel(labels[i])
        ax[i].set_xticks(xticks)

    plt.savefig('/home/gpruto/metal_ab/images/redshift_evolution/violinplot_%s.png' % gal, dpi=300, bbox_inches='tight')



### with weights!
else:
    if test_for_allgal: #run for g33206 and g10304 together just to test weather the code is okay
        targethalo2 = TargetHalo('g10304', run)
        
        for i,snap in enumerate(snaps):
            si_o, c_o, c_fe, o_fe, si_fe, c_si, redshift, volume = abundances_for_snap(targethalo2, snap)
            #append all data to the previous ones
            si_o_allz[i] = np.append(si_o_allz[i], si_o)
            c_o_allz[i] = np.append(c_o_allz[i], c_o)
            c_fe_allz[i] = np.append(c_fe_allz[i], c_fe)
            o_fe_allz[i] = np.append(o_fe_allz[i], o_fe)
            si_fe_allz[i] = np.append(si_fe_allz[i], si_fe)
            c_si_allz[i] = np.append(c_si_allz[i], c_si)
            volume_allz[i] = np.append(volume_allz[i], volume)
            
        data = [c_si_allz, c_o_allz, c_fe_allz, o_fe_allz, si_fe_allz, si_o_allz]

        for i in range(len(ax)):
            for j in range(len(data[i])):
                density, W, p16, p84, median = weighted_kde(data[i][j], volume_allz[j], intervals[i], bw=0.3)
                plot_violin(ax[i], intervals[i], density, p16, p84, median, x=reds[j], color=color_violin, alpha=0.3)
                xticks = [2,3,4,5,6,7,8,9]
                ax[i].set_xlabel('Redshift')
                ax[i].set_ylabel(labels[i])
                ax[i].set_xticks(xticks)

        plt.savefig('/home/gpruto/metal_ab/images/redshift_evolution/violinplot_2gal_%s_weighted.png' % gal, dpi=300, bbox_inches='tight')

    

    else:
        writing = np.zeros((len(ax), len(reds), 500)) #redshift, abundance, density
        tot_weights = np.zeros((len(ax), len(reds)))
        for i in range(len(ax)):
            for j in range(len(data[i])):
                if len(data[i][j]) > 0:
                    density, W, p16, p84, median = weighted_kde(data[i][j], volume_allz[j], intervals[i], bw=0.3)
                    plot_violin(ax[i], intervals[i], density, p16, p84, median, x=reds[j], color=color_violin, alpha=0.3)
                    #weighted_violin(ax[i], (data[i][j]), (volume_allz[j]), x=reds[j], width=0.4, color=color_violin, alpha=0.3)
                    xticks = [2,3,4,5,6,7,8,9]
                    ax[i].set_xlabel('Redshift')
                    ax[i].set_ylabel(labels[i])
                    ax[i].set_xticks(xticks)

                    writing[i][j] = density
                    tot_weights[i][j] = W
        
        for j in range(len(reds)):
            outfile = '/home/gpruto/metal_ab/code/redshift_evolution/violin_plots/z=%d/%s/violinplot_data.txt' % (reds[j], gal)
            with open(outfile, 'w') as f:
                f.write('#[C/Si]\t[C/O]\t[C/Fe]\t[O/Fe]\t[Si/Fe]\t[Si/O]\n')
                f.write('Total weights: %s\n' % ','.join(map(str, tot_weights[:,j])))
                for k in range(len(writing[0][j])):
                    f.write(f"{writing[0][j][k]}\t{writing[1][j][k]}\t{writing[2][j][k]}\t"f"{writing[3][j][k]}\t{writing[4][j][k]}\t{writing[5][j][k]}\n")
        plt.savefig('/home/gpruto/metal_ab/images/redshift_evolution/violinplot_%s_weighted.png' % gal, dpi=300, bbox_inches='tight')

