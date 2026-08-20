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



from_single_violins = False #if true it starts from distributions of single galaxies, if false it reads DLA_cut files
neutral_frac_min = 0.1
density_lim = -2
temp_lim = np.log10(2e4)
metallicity_lim = -4

redshift_intervals = [2, 4.5, 6.5]
grey_region_mean = [mlib.c_si_sod_mean, mlib.c_o_sod_mean, mlib.c_fe_sod_mean, mlib.o_fe_sod_mean, mlib.si_fe_sod_mean, mlib.si_o_sod_mean]
grey_region_sigma_mean = [mlib.c_si_sod_mean_err, mlib.c_o_sod_mean_err, mlib.c_fe_sod_mean_err, mlib.o_fe_sod_mean_err, mlib.si_fe_sod_mean_err, mlib.si_o_sod_mean_err]
grey_region_sigma = [mlib.c_si_sod_sigma, mlib.c_o_sod_sigma, mlib.c_fe_sod_sigma, mlib.o_fe_sod_sigma, mlib.si_fe_sod_sigma, mlib.si_o_sod_sigma]
        

def plot_violin(ax, y, density,  x=0, color='C0', alpha=0.6):
    #normalise such that the maximum width is 0.4
    density = density / np.max(density) * 0.36
    ax.fill_betweenx(y, x - density, x + density, color=color, alpha=alpha)


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



snaps = [188, 129, 92, 68, 51, 39, 30]
reds = [3,4, 5,6,7,8, 9]
colors = cmc.batlowK(np.linspace(0, 1, 14))
gals = ['g5229300', 'g2274036', 'g519761', 'g500531', 'g137030', 'g37591','g33206', 'g10304', 'g5760', 'g1163', 'g578', 'g205', 'g39']#, 'g2']


if from_single_violins:
    abundance_density = np.zeros((len(reds), 6, 500)) #redshift, abundance ratio, distribution bins

    #read all the violins and weight them depending on the number of particles
    for z in range(len(reds)):
        total_weight = np.zeros(6)
        for g in range(len(gals)):
            infile = '/home/gpruto/metal_ab/code/redshift_evolution/violin_plots/z=%d/%s/violinplot_data.txt' #redshift, galaxy
            with open(infile % (reds[z], gals[g]), 'r') as f:
                f.readline() #skip header
                dist_weight = (f.readline()[15:-2].split(','))
                dist_weight = np.array(dist_weight, dtype=float)
                total_weight += dist_weight
                data_abundances = np.loadtxt(f)
                data_abundances = data_abundances.T
                
                for k in range(6):
                    abundance_density[z][k] += dist_weight[k]*data_abundances[k] #fix for weights)
                
        abundance_density[z] /= np.sum(total_weight) #normalise by total weight

    fig, ax = plt.subplots(3,2, figsize = (12, 15))
    ax = ax.flatten()

    labels = ['[C/Si]', '[C/O]', '[C/Fe]', '[O/Fe]', '[Si/Fe]', '[Si/O]']
    intervals = [np.linspace(-0.4, 0.5, 500), np.linspace(-1, 0.2, 500),
                    np.linspace(-0.3, 1, 500), np.linspace(0, 1.7, 500),
                    np.linspace(-0.2, 0.8, 500), np.linspace(-1.2, 0.4, 500)]


    for i in range(len(ax)):
        for j in range(2):
            ax[i].fill_betweenx((grey_region_mean[i][j]-grey_region_sigma[i][j], grey_region_mean[i][j]+grey_region_sigma[i][j]), x1=redshift_intervals[j], x2=redshift_intervals[j+1], color='grey', alpha=0.3, label='observations')
            ax[i].fill_betweenx((grey_region_mean[i][j]-grey_region_sigma_mean[i][j], grey_region_mean[i][j]+grey_region_sigma_mean[i][j]), x1=redshift_intervals[j], x2=redshift_intervals[j+1], color='grey', alpha=0.8)


    for i in range(len(ax)):
        for j in range(len(reds)):
            plot_violin(ax[i], intervals[i], abundance_density[j][i], x=reds[j], color='royalblue', alpha=0.3)

            #plotting median and 16th and 84th percentile lines
            total_ab = 0
            p16 = -100
            p84 = -100
            median = -100
            abundance_density[j][i] /= np.sum(abundance_density[j][i]) #normalise the distribution for this abundance ratio and redshift
            for k in range(len(intervals[i])):
                total_ab += abundance_density[j][i][k]
                if total_ab >= 0.16 and p16 == -100:
                    p16 = intervals[i][k]
                if total_ab >= 0.84 and p84 == -100:
                    p84 = intervals[i][k]
                if total_ab >= 0.5 and median == -100:
                    median = intervals[i][k]


            ax[i].hlines(median, reds[j]-0.3, reds[j]+0.3, color='royalblue', lw=1.5)
            ax[i].hlines(p16, reds[j]-0.3, reds[j]+0.3, color='royalblue', lw=1.5)
            ax[i].hlines(p84, reds[j]-0.3, reds[j]+0.3, color='royalblue', lw=1.5)
            ax[i].vlines(reds[j], p16, p84, color='royalblue', lw=1.5)

            ax[i].set_xlabel('Redshift')
            ax[i].set_ylabel(labels[i])
            ax[i].set_xticks([2,3,4,5,6,7,8,9])


    fig.savefig('/home/gpruto/metal_ab/images/redshift_evolution/violinplot_weighted_allgal.png', dpi=300, bbox_inches='tight')
    fig.savefig('/home/gpruto/metal_ab/images/paper/violinplot_weighted_allgal.png', dpi=300, bbox_inches='tight')


else:
    labels = ['[C/Si]', '[C/O]', '[C/Fe]', '[O/Fe]', '[Si/Fe]', '[Si/O]']
    #f.write('#x y z mass volume n_H n_HI [C/O] [Si/O] [C/Fe] [O/Fe] [Si/Fe] [Si/C]\n')
    c_si = [[] for _ in range(len(reds))]
    c_o = [[] for _ in range(len(reds))]
    c_fe = [[] for _ in range(len(reds))]
    o_fe = [[] for _ in range(len(reds))]
    si_fe = [[] for _ in range(len(reds))]
    si_o = [[] for _ in range(len(reds))]
    volume = [[] for _ in range(len(reds))]

    for z in range(len(reds)):
        for g in range(len(gals)):
            infile = '/home/gpruto/metal_ab/code/DLA_cut/z=%d/%s/dla_cuts_x_HI>%.1f_n_H>%d_T<%.1f_met>%.1f.txt' %(reds[z], gals[g], neutral_frac_min, density_lim, temp_lim, metallicity_lim)
            with open(infile, 'r') as f:
                data = np.loadtxt(f, skiprows=1, usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12), unpack=True)
                data = data.T
                print(data.shape)
                if (len(data)>0):
                    volume[z].extend(data[:, 4].tolist())
                    c_si[z].extend((-data[:, 12]).tolist())
                    c_o[z].extend(data[:, 7].tolist())
                    si_o[z].extend(data[:, 8].tolist())
                    c_fe[z].extend(data[:, 9].tolist())
                    o_fe[z].extend(data[:, 10].tolist())
                    si_fe[z].extend(data[:, 11].tolist())

    print(len(volume), len(volume[0]))
    fig, ax = plt.subplots(3,2, figsize = (12, 15))
    ax = ax.flatten()

    abundances = [c_si, c_o, c_fe, o_fe, si_fe, si_o]
    for i in range(len(abundances)):
        for z in range(len(reds)):
            density, _, p16, p84, median = weighted_kde(abundances[i][z], volume[z], np.linspace(np.min(abundances[i][z]), np.max(abundances[i][z]), 100), bw = 0.25)
            plot_violin(ax[i], np.linspace(np.min(abundances[i][z]), np.max(abundances[i][z]), 100), density, x=reds[z], color='royalblue', alpha=0.3)
            ax[i].hlines(median, reds[z]-0.25, reds[z]+0.25, color='royalblue', lw=1.5)
            ax[i].hlines(p16, reds[z]-0.25, reds[z]+0.25, color='royalblue', lw=1.5)
            ax[i].hlines(p84, reds[z]-0.25, reds[z]+0.25, color='royalblue', lw=1.5)
            ax[i].vlines(reds[z], p16, p84, color='royalblue', lw=1.5)
       
        for j in range(2):
            ax[i].fill_betweenx((grey_region_mean[i][j]-grey_region_sigma[i][j], grey_region_mean[i][j]+grey_region_sigma[i][j]), x1=redshift_intervals[j], x2=redshift_intervals[j+1], color='grey', alpha=0.3, label='observations')
            ax[i].fill_betweenx((grey_region_mean[i][j]-grey_region_sigma_mean[i][j], grey_region_mean[i][j]+grey_region_sigma_mean[i][j]), x1=redshift_intervals[j], x2=redshift_intervals[j+1], color='grey', alpha=0.8)
        
        xticks = [2,3,4,5,6,7,8,9]
        ax[i].set_xlabel('Redshift')
        ax[i].set_ylabel(labels[i])
        ax[i].set_xticks(xticks)


    fig.savefig('/home/gpruto/metal_ab/images/redshift_evolution/violinplot_weighted_allgal_method2.png', dpi=300, bbox_inches='tight')
    fig.savefig('/home/gpruto/metal_ab/images/paper/violinplot_weighted_allgal_method2.png', dpi=300, bbox_inches='tight')
        

        
                    


    

        





