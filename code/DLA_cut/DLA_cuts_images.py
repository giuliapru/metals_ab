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
from scipy.spatial import cKDTree

#### SNAP
snap = int(sys.argv[1]) #188, 129, 92, 68, 51
if snap == 188:
    red = 3
if snap == 129:
    red = 4
if snap == 92:
    red = 5
elif snap == 68:
    red = 6
elif snap == 51:
    red = 7
elif snap == 39:
    red = 8

def add_cbar(fig, im, ax, label, color, off_left = 0.025, off_bottom = 0.005, orientation='horizontal', label_pos='top', label_color = ''):
    if label_color == '':
        label_color = color

    # same layout for every colorbar
    cax = fig.add_axes([
        ax.get_position().x0 + off_left,   
        ax.get_position().y0 + off_bottom,        
        ax.get_position().x1 - ax.get_position().x0 - 2*off_left,             
        0.018 ])
    cbar = fig.colorbar(im, cax=cax, orientation=orientation)
    cbar.set_label(label, color = label_color, fontsize=14)  
    cbar.ax.xaxis.set_tick_params(color= color, labelcolor= color)
    cbar.ax.tick_params(color= color, labelcolor= color)
    cbar.outline.set_edgecolor(color)
    cbar.ax.xaxis.set_ticks_position(label_pos)
    cbar.ax.xaxis.set_label_position(label_pos)


#### PARAMETERS
run = 'fiducial'
cond_hr = 0.5
gal = str(sys.argv[2])
hist = False
plots2x2 = False
plots1x3 = False
maps = False #(gal=='g39') or (gal=='g205')
maps_combined = True
writefile = False
Z_solar = 0.0127

neutral_frac_min = 0.1 #minimum neutral fraction in the gas particle
temp_lim = np.log10(2e4) #maximum temperature
density_lim = -2 #minimum density
metallicity_lim = -4 #minimum metallicity
###############


#### READ DATA
targethalo = TargetHalo(gal, run)
targethalo.read_haloes(snap, 0)

coords, volume, redshift, gasmass, _, _, h_density, hi_density, carbon_density, oxygen_density, silicon_density, iron_density, temperature, metallicity = targethalo.gas_properties(snap, cond_hr, 1., all=True, metals=True, temperature=True, metallicity=True)
cond_neutral = (hi_density/h_density) > neutral_frac_min
dla_cond = ((hi_density/h_density) > neutral_frac_min) & (np.log10(h_density) > density_lim) & (temperature < 10**temp_lim) & (np.log10(metallicity/Z_solar) > metallicity_lim)

print('The fraction of gas particles that are DLA is: ', np.sum(dla_cond)/len(dla_cond))
dla_vol_frac = np.sum(volume[dla_cond])/np.sum(volume)
print('The volume fraction of gas that is DLA is: ', dla_vol_frac)

'''#### NEUTRAL FRACTION VS DENSITY AND TEMPERATURE AND METALLICITY
if hist==True and plots2x2==True:
    #fig, ax = plt.subplots(2, 2, figsize=(10,10))
    #ax = ax.flatten()
    fig, ax = plt.subplots(1,4, figsize=(20,5))

    x_hi_bins = np.linspace(0,1, 150)
    n_h_bins = np.linspace(-6, 5, 300)
    T_bins = np.linspace(1, 7.5, 300)
    met_bins = np.linspace(-15, 1, 300)

    mlib.hist_2d(np.log10(h_density), (hi_density/h_density), ax[0], x_bins = n_h_bins, y_bins=x_hi_bins, alpha=0.3)
    mlib.hist_2d(np.log10(temperature), (hi_density/h_density), ax[1], x_bins = T_bins, y_bins=x_hi_bins, alpha=0.3)
    mlib.hist_2d(np.log10(metallicity/Z_solar), (hi_density/h_density), ax[2], x_bins = met_bins, y_bins=x_hi_bins, alpha=0.3)
    mlib.hist_2d(np.log10(h_density), np.log10(temperature), ax[3], x_bins = n_h_bins, y_bins=T_bins, alpha=0.3)
    
    mlib.hist_2d(np.log10(h_density[dla_cond]), (hi_density[dla_cond]/h_density[dla_cond]), ax[0], x_bins = n_h_bins, y_bins=x_hi_bins)
    mlib.hist_2d(np.log10(temperature[dla_cond]), (hi_density[dla_cond]/h_density[dla_cond]), ax[1], x_bins = T_bins, y_bins=x_hi_bins)
    mlib.hist_2d( np.log10(metallicity[dla_cond]/Z_solar), (hi_density[dla_cond]/h_density[dla_cond]), ax[2], x_bins = met_bins, y_bins=x_hi_bins)
    mlib.hist_2d(np.log10(h_density[dla_cond]), np.log10(temperature[dla_cond]), ax[3], x_bins = n_h_bins, y_bins=T_bins)
    

    ax[0].plot([-6, 5], [0.1, 0.1], color='k', ls='--', lw=2)
    ax[0].vlines(density_lim, 0, 1, color='k', ls='--', lw=2)

    ax[1].plot([1, 7.5], [0.1, 0.1], color='k', ls='--', lw=2)
    ax[1].vlines(temp_lim, 0, 1, color='k', ls='--', lw=2)

    ax[2].plot([-15, 1], [0.1, 0.1], color='k', ls='--', lw=2)
    ax[2].vlines(metallicity_lim, 0, 1, color='k', ls='--', lw=2)

    ax[3].plot([-6, 5], [temp_lim, temp_lim], color='k', ls='--', lw=2)
    ax[3].plot([density_lim, density_lim], [1, 7.5], color='k', ls='--', lw=2)

    ax[0].set_xlabel(r'$\log_{10} (n_{\rm H}$ [cm$^{-3}$])')
    ax[1].set_xlabel(r'$\log_{10} (T$ [K])')
    ax[2].set_xlabel(r'$\log_{10} (Z/Z_{\odot}$)')
    ax[3].set_xlabel(r'$\log_{10} (n_{\rm H}$ [cm$^{-3}$])')

    ax[0].set_ylabel(r'$n_{\rm HI}/n_{\rm H}$')
    ax[1].set_ylabel(r'$n_{\rm HI}/n_{\rm H}$')
    ax[2].set_ylabel(r'$n_{\rm HI}/n_{\rm H}$')
    ax[3].set_ylabel(r'$\log_{10} (T$ [K])')

    ax[0].set_ylim(-0.01,1.01)
    ax[1].set_ylim(-0.01,1.01)
    ax[2].set_ylim(-0.01,1.01)
    ax[3].set_ylim(1, 7.5)
   
    ax[0].set_xlim(-6,5)
    ax[1].set_xlim(1,7.5)
    ax[2].set_xlim(-15,1)
    ax[3].set_xlim(-6,5)
 
    #fig.savefig('/home/gpruto/metal_ab/images/paper/dla_cuts_%s_z%d_2x2.png' %(gal, redshift), bbox_inches='tight', dpi=300)
    fig.savefig('/home/gpruto/metal_ab/images/paper/dla_cuts_%s_z%d_1x4.png' %(gal, redshift), bbox_inches='tight', dpi=300)


if hist==True and plots1x3==True:
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax = ax.flatten()

    x_hi_bins = np.linspace(0,1, 150)
    n_h_bins = np.linspace(-6, 5, 300)
    T_bins = np.linspace(1, 7.5, 300)
    met_bins = np.linspace(-15, 1, 300)

    mlib.hist_2d(np.log10(h_density), (hi_density/h_density), ax[0], x_bins = n_h_bins, y_bins=x_hi_bins, alpha=0.3)
    mlib.hist_2d(np.log10(temperature), (hi_density/h_density), ax[1], x_bins = T_bins, y_bins=x_hi_bins, alpha=0.3)
    mlib.hist_2d(np.log10(metallicity/Z_solar), (hi_density/h_density), ax[2], x_bins = met_bins, y_bins=x_hi_bins, alpha=0.3)

    mlib.hist_2d(np.log10(h_density[dla_cond]), (hi_density[dla_cond]/h_density[dla_cond]), ax[0], x_bins = n_h_bins, y_bins=x_hi_bins)
    mlib.hist_2d(np.log10(temperature[dla_cond]), (hi_density[dla_cond]/h_density[dla_cond]), ax[1], x_bins = T_bins, y_bins=x_hi_bins)
    mlib.hist_2d(np.log10(metallicity[dla_cond]/Z_solar), (hi_density[dla_cond]/h_density[dla_cond]), ax[2], x_bins = met_bins, y_bins=x_hi_bins)

    ax[0].plot([-6, 5], [0.1, 0.1], color='k', ls='--', lw=2)
    ax[0].vlines(density_lim, 0, 1, color='k', ls='--', lw=2)

    ax[1].plot([1, 7.5], [0.1, 0.1], color='k', ls='--', lw=2)
    ax[1].vlines(temp_lim, 0, 1, color='k', ls='--', lw=2)

    ax[2].plot([-15, 1], [0.1, 0.1], color='k', ls='--', lw=2)
    ax[2].vlines(metallicity_lim, 0, 1, color='k', ls='--', lw=2)

    ax[0].set_xlabel(r'$\log_{10} (n_{\rm H}$ [cm$^{-3}$])')
    ax[1].set_xlabel(r'$\log_{10} (T$ [K])')
    ax[2].set_xlabel(r'$\log_{10} (Z/Z_{\odot}$)')
    
    ax[0].set_ylabel(r'$n_{\rm HI}/n_{\rm H}$')
    ax[1].set_ylabel(r'$n_{\rm HI}/n_{\rm H}$')
    ax[2].set_ylabel(r'$n_{\rm HI}/n_{\rm H}$')

    ax[0].set_ylim(-0.01,1.01)
    ax[1].set_ylim(-0.01,1.01)
    ax[2].set_ylim(-0.01,1.01)

    ax[0].set_xlim(-6,5)
    ax[1].set_xlim(1,7.5)
    ax[2].set_xlim(-15,1)

    fig.savefig('/home/gpruto/metal_ab/images/paper/dla_cuts_%s_z%d_1x3.png' %(gal, redshift), bbox_inches='tight', dpi=300)

'''

### TEST METALLICITY and FE/H
'''fig_z, ax_z = plt.subplots(1,3, figsize=(18,5))
mlib.hist_2d(np.log10(metallicity/Z_solar), np.log10(h_density), ax_z[0], x_bins = 300, y_bins=300, alpha=0.3)
mlib.hist_2d(np.log10(metallicity[dla_cond]/Z_solar), np.log10(h_density[dla_cond]), ax_z[0], x_bins = 300, y_bins=300)

mlib.hist_2d(np.log10(iron_density) - mlib.Fe_sun, np.log10(h_density), ax_z[1], x_bins = 300, y_bins=300, alpha=0.3)
mlib.hist_2d(np.log10(iron_density[dla_cond]) - mlib.Fe_sun, np.log10(h_density[dla_cond]), ax_z[1],x_bins = 300, y_bins=300)

mlib.hist_2d(np.log10(metallicity/Z_solar), np.log10(iron_density) - mlib.Fe_sun, ax_z[2], x_bins = 300, y_bins=300, alpha=0.3)
mlib.hist_2d(np.log10(metallicity[dla_cond]/Z_solar), np.log10(iron_density[dla_cond]) - mlib.Fe_sun, ax_z[2], x_bins = 300, y_bins=300)  
ax_z[2].plot([-12, -2], [-12, -2], color='k', ls='--', lw=2)

ax_z[0].set_ylabel(r'$\log_{10} (n_{\rm H}$ [cm$^{-3}$])')
ax_z[0].set_xlabel(r'$\log_{10} (Z/Z_{\odot}$)')
ax_z[1].set_ylabel(r'$\log_{10} (n_{\rm H}$ [cm$^{-3}$])')
ax_z[1].set_xlabel(r'[Fe/H]')
ax_z[2].set_xlabel(r'$\log_{10} (Z/Z_{\odot}$)')
ax_z[2].set_ylabel(r'[Fe/H]')

fig_z.savefig('/home/gpruto/metal_ab/images/paper/dla_cuts_metallicity_%s_z%d.png' %(gal, redshift), bbox_inches='tight', dpi=300)
'''

#### MAPS of DLAs
halocoord, mass200, massDM, halomass, haloradius, redshift = lib.read_halos(gal, run, snap)
centre = halocoord[0]
print('The target mass is %e' % (halomass[0]*1e10/lib.h))
'''
if maps:
    fig_m, ax_m = plt.subplots(2,2, figsize=(12,12))
    ax_m = ax_m.flatten()
    plt.subplots_adjust(wspace=0., hspace=0.)

    z_thick = 300
    xy_width = 300
    z_slice_haloes = (halocoord[:, 2]>(centre[2]-z_thick)) & (halocoord[:, 2]<(centre[2]+z_thick)) & (halocoord[:,0]>(centre[0]-xy_width)) & (halocoord[:,0]<(centre[0]+xy_width)) & (halocoord[:,1]>(centre[1]-xy_width)) & (halocoord[:,1]<(centre[1]+xy_width))
    z_slice = (coords[:,  2]>(centre[2]-z_thick)) & (coords[:,  2]<(centre[2]+z_thick)) & (coords[:,0]>(centre[0]-xy_width)) & (coords[:,0]<(centre[0]+xy_width)) & (coords[:,1]>(centre[1]-xy_width)) & (coords[:,1]<(centre[1]+xy_width))
    z_slice_dla = dla_cond & (coords[:,  2]>(centre[2]-z_thick)) & (coords[:,  2]<(centre[2]+z_thick)) & (coords[:,0]>(centre[0]-xy_width)) & (coords[:,0]<(centre[0]+xy_width)) & (coords[:,1]>(centre[1]-xy_width)) & (coords[:,1]<(centre[1]+xy_width))

    #### Selected DLA cells
    color_dla = '#f9ad95'
    ax_m[0].scatter(coords[z_slice, 0], coords[z_slice, 1], s=0.0006, c='black')
    ax_m[0].scatter(coords[z_slice_dla, 0], coords[z_slice_dla, 1], s=0.0006, c=color_dla)
    ax_m[0].scatter([], [], c='black', label='All gas')
    ax_m[0].scatter([], [], c=color_dla, label='DLA selected')

    color_vr =  '#fae3d4'
    angle = np.linspace(0, 2*np.pi, 100)
    for i in range(len(halocoord[z_slice_haloes])):
        if halomass[z_slice_haloes][i] > 1e8/(1e10/lib.h):
            x = halocoord[z_slice_haloes][i, 0] + haloradius[z_slice_haloes][i]*np.cos(angle)
            y = halocoord[z_slice_haloes][i, 1] + haloradius[z_slice_haloes][i]*np.sin(angle)
            ax_m[0].plot(x, y, color=color_vr, lw=1)
    ax_m[0].set_xlim(centre[0]-xy_width, centre[0]+xy_width)
    ax_m[0].set_ylim(centre[1]-xy_width, centre[1]+xy_width)
    ax_m[0].legend(frameon=True, framealpha=0.5, fontsize=13)

        

    #### SLICES
    h_density = h_density[z_slice]
    temperature = temperature[z_slice]
    metallicity = metallicity[z_slice]
    gasmass = gasmass[z_slice]
    oxygen_density = oxygen_density[z_slice]
    iron_density = iron_density[z_slice]
    carbon_density = carbon_density[z_slice]
    silicon_density = silicon_density[z_slice]

    PlotSize=300
    Nplot=300
    centre = targethalo.data[snap]['coord']
    gas_coord_centred = coords[z_slice] - centre
    Edges1d = np.linspace(-PlotSize, PlotSize, Nplot+1, endpoint=True, dtype=np.float64)
    Grid1d = 0.5*(Edges1d[1:] + Edges1d[:-1])

    xgrid, ygrid, zgrid = np.meshgrid(Grid1d, Grid1d, Grid1d)
    Grid3D = np.array( [xgrid.reshape(Nplot**3), ygrid.reshape(Nplot**3), zgrid.reshape(Nplot**3)]).T
    _, cells = spatial.KDTree(np.array(gas_coord_centred[:,:])).query( Grid3D, k=1 )
    weight_slice = (gasmass[cells]).reshape((Nplot,Nplot,Nplot))

    density_slice = (h_density[cells]).reshape((Nplot,Nplot,Nplot))
    density_slice = np.sum(density_slice*weight_slice, axis=2)/np.sum(weight_slice, axis=2)

    metallicity_slice = (metallicity[cells]).reshape((Nplot,Nplot,Nplot))
    metallicity_slice = np.sum(metallicity_slice*weight_slice, axis=2)/np.sum(weight_slice, axis=2)

    temperature_slice = (temperature[cells]).reshape((Nplot,Nplot,Nplot))
    temperature_slice = np.sum(temperature_slice*weight_slice, axis=2)/np.sum(weight_slice, axis=2)

    vmin_dens = np.percentile(np.log10(density_slice),2)
    vmax_dens = np.percentile(np.log10(density_slice),98)
    im0 = ax_m[1].imshow(np.log10(density_slice), origin='lower', extent=(-PlotSize, PlotSize, -PlotSize, PlotSize), cmap=cmc.lajolla, vmin=vmin_dens, vmax=vmax_dens)
    add_cbar(fig_m, im0, ax_m[1], label=r'log ($n_{\rm H}$ [cm$^{-3}$])', color='white')

    vmin_metal = np.percentile(np.log10(metallicity_slice/Z_solar),2)
    vmax_metal = np.percentile(np.log10(metallicity_slice/Z_solar),98)
    im1 = ax_m[2].imshow(np.log10(metallicity_slice/Z_solar), origin='lower', extent=(-PlotSize, PlotSize, -PlotSize, PlotSize), cmap=cmc.acton, vmin=vmin_metal, vmax=vmax_metal)
    add_cbar(fig_m, im1, ax_m[2], label=r'log ($Z/Z_{\odot}$)', color='white')

    vmin_temp = np.percentile(np.log10(temperature_slice),2)
    vmax_temp = np.percentile(np.log10(temperature_slice),98)
    im2 = ax_m[3].imshow(np.log10(temperature_slice), origin='lower', extent=(-PlotSize, PlotSize, -PlotSize, PlotSize), cmap=cmc.lipari, vmin=vmin_temp, vmax=vmax_temp)
    add_cbar(fig_m, im2, ax_m[3], label=r'log (T [K])', color='white')  

    for j in range(len(ax_m)-1):
        for i in range(len(halocoord[z_slice_haloes])):
            if halomass[z_slice_haloes][i] > 1e8/(1e10/lib.h):
                x = halocoord[z_slice_haloes][i, 0] + haloradius[z_slice_haloes][i]*np.cos(angle) - centre[0]
                y = halocoord[z_slice_haloes][i, 1] + haloradius[z_slice_haloes][i]*np.sin(angle) - centre[1]
                ax_m[j+1].plot(x, y, color=color_vr, lw=1)

    for j in range(len(ax_m)):
        ax_m[j].set_xticks([])
        ax_m[j].set_yticks([])
        ax_m[j].set_xlabel('')
        ax_m[j].set_ylabel('')
        if j >0:
            ax_m[j].set_xlim(-PlotSize, PlotSize)
            ax_m[j].set_ylim(-PlotSize, PlotSize)
        
    fig_m.savefig('/home/gpruto/metal_ab/images/paper/dla_cuts_map_%s_z%d.png' %(gal, redshift), bbox_inches='tight', dpi=300)


    ##### ABUNDANCE MAPS
    fig_ab, ax_ab = plt.subplots(2, 3, figsize=(18,12))
    ax_ab = ax_ab.flatten()
    plt.subplots_adjust(wspace=0., hspace=0.)

    o_fe_slice = (oxygen_density[cells]/iron_density[cells]).reshape((Nplot,Nplot,Nplot))
    o_fe_slice = np.sum(o_fe_slice*weight_slice, axis=2)/np.sum(weight_slice, axis=2)
    o_fe_slice = np.log10(o_fe_slice) - mlib.O_Fe_solar
    vmin_o_fe = np.percentile(o_fe_slice,2)
    vmax_o_fe = np.percentile(o_fe_slice,98)
    im0 = ax_ab[0].imshow(o_fe_slice, origin='lower', extent=(-PlotSize, PlotSize, -PlotSize, PlotSize), cmap=cmc.lajolla, vmin=vmin_o_fe, vmax=vmax_o_fe)
    add_cbar(fig_ab, im0, ax_ab[0], label=r'[O/Fe]', color='gray')

    c_fe_slice = (carbon_density[cells]/iron_density[cells]).reshape((Nplot,Nplot,Nplot))
    c_fe_slice = np.sum(c_fe_slice*weight_slice, axis=2)/np.sum(weight_slice, axis=2)
    c_fe_slice = np.log10(c_fe_slice) - mlib.C_Fe_solar
    vmin_c_fe = np.percentile(c_fe_slice,2)
    vmax_c_fe = np.percentile(c_fe_slice,98)
    im1 = ax_ab[1].imshow(c_fe_slice, origin='lower', extent=(-PlotSize, PlotSize, -PlotSize, PlotSize), cmap=cmc.lipari, vmin=vmin_c_fe, vmax=vmax_c_fe)
    add_cbar(fig_ab, im1, ax_ab[1], label=r'[C/Fe]', color='gray')

    si_c_slice = (silicon_density[cells]/carbon_density[cells]).reshape((Nplot,Nplot,Nplot))
    si_c_slice = np.sum(si_c_slice*weight_slice, axis=2)/np.sum(weight_slice, axis=2)
    si_c_slice = np.log10(si_c_slice) - mlib.Si_C_solar
    vmin_si_c = np.percentile(si_c_slice,2)
    vmax_si_c = np.percentile(si_c_slice,98)
    im2 = ax_ab[2].imshow(si_c_slice, origin='lower', extent=(-PlotSize, PlotSize, -PlotSize, PlotSize), cmap=cmc.navia, vmin=vmin_si_c, vmax=vmax_si_c)
    add_cbar(fig_ab, im2, ax_ab[2], label=r'[Si/C]', color='black')

    si_fe_slice = (silicon_density[cells]/iron_density[cells]).reshape((Nplot,Nplot,Nplot))
    si_fe_slice = np.sum(si_fe_slice*weight_slice, axis=2)/np.sum(weight_slice, axis=2)
    si_fe_slice = np.log10(si_fe_slice) - mlib.Si_Fe_solar
    vmin_si_fe = np.percentile(si_fe_slice,2)
    vmax_si_fe = np.percentile(si_fe_slice,98)
    im3 = ax_ab[3].imshow(si_fe_slice, origin='lower', extent=(-PlotSize, PlotSize, -PlotSize, PlotSize), cmap=cmc.acton, vmin=vmin_si_fe, vmax=vmax_si_fe)
    add_cbar(fig_ab, im3, ax_ab[3], label=r'[Si/Fe]', color='black')

    si_o_slice = (silicon_density[cells]/oxygen_density[cells]).reshape((Nplot,Nplot,Nplot))
    si_o_slice = np.sum(si_o_slice*weight_slice, axis=2)/np.sum(weight_slice, axis=2)
    si_o_slice = np.log10(si_o_slice) - mlib.Si_O_solar
    vmin_si_o = np.percentile(si_o_slice,2)
    vmax_si_o = np.percentile(si_o_slice,98)
    im4 = ax_ab[4].imshow(si_o_slice, origin='lower', extent=(-PlotSize, PlotSize, -PlotSize, PlotSize), cmap=cmc.lapaz, vmin=vmin_si_o, vmax=vmax_si_o)
    add_cbar(fig_ab, im4, ax_ab[4], label=r'[Si/O]', color='black')

    c_o_slice = (carbon_density[cells]/oxygen_density[cells]).reshape((Nplot,Nplot,Nplot))
    c_o_slice = np.sum(c_o_slice*weight_slice, axis=2)/np.sum(weight_slice, axis=2)
    c_o_slice = np.log10(c_o_slice) - mlib.C_O_solar
    vmin_c_o = np.percentile(c_o_slice,2)
    vmax_c_o = np.percentile(c_o_slice,98)
    im5 = ax_ab[5].imshow(c_o_slice, origin='lower', extent=(-PlotSize, PlotSize, -PlotSize, PlotSize), cmap=cmc.devon, vmin=vmin_c_o, vmax=vmax_c_o)
    add_cbar(fig_ab, im5, ax_ab[5], label=r'[C/O]', color='gray')

    for j in range(len(ax_ab)):
        for i in range(len(halocoord[z_slice_haloes])):
            if halomass[z_slice_haloes][i] > 1e8/(1e10/lib.h):
                x = halocoord[z_slice_haloes][i, 0] + haloradius[z_slice_haloes][i]*np.cos(angle) - centre[0]
                y = halocoord[z_slice_haloes][i, 1] + haloradius[z_slice_haloes][i]*np.sin(angle) - centre[1]
                ax_ab[j].plot(x, y, color=color_vr, lw=1.5)

    for j in range(len(ax_ab)):
        ax_ab[j].set_xticks([])
        ax_ab[j].set_yticks([])
        ax_ab[j].set_xlabel('')
        ax_ab[j].set_ylabel('')
        ax_ab[j].set_xlim(-PlotSize, PlotSize)
        ax_ab[j].set_ylim(-PlotSize, PlotSize)

    fig_ab.savefig('/home/gpruto/metal_ab/images/paper/dla_cuts_ab_map_%s_z%d.png' %(gal, redshift), bbox_inches='tight', dpi=300)
'''
if maps_combined == True:
    fig, ax = plt.subplots(3,3, figsize=(18,18))
    ax = ax.flatten()
    plt.subplots_adjust(wspace=0., hspace=0.)
    for a in ax:
        for spine in a.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('white')

    z_thick = 300
    xy_width = 300
    z_slice_haloes = (halocoord[:, 2]>(centre[2]-z_thick)) & (halocoord[:, 2]<(centre[2]+z_thick)) & (halocoord[:,0]>(centre[0]-xy_width)) & (halocoord[:,0]<(centre[0]+xy_width)) & (halocoord[:,1]>(centre[1]-xy_width)) & (halocoord[:,1]<(centre[1]+xy_width))
    z_slice = (coords[:,  2]>(centre[2]-z_thick)) & (coords[:,  2]<(centre[2]+z_thick)) & (coords[:,0]>(centre[0]-xy_width)) & (coords[:,0]<(centre[0]+xy_width)) & (coords[:,1]>(centre[1]-xy_width)) & (coords[:,1]<(centre[1]+xy_width))
    z_slice_dla = dla_cond & (coords[:,  2]>(centre[2]-z_thick)) & (coords[:,  2]<(centre[2]+z_thick)) & (coords[:,0]>(centre[0]-xy_width)) & (coords[:,0]<(centre[0]+xy_width)) & (coords[:,1]>(centre[1]-xy_width)) & (coords[:,1]<(centre[1]+xy_width))

    #### Selected DLA cells
    color_dla = '#f9ad95'
    ax[0].scatter(coords[z_slice, 0], coords[z_slice, 1], s=0.0006, c='black')
    ax[0].scatter(coords[z_slice_dla, 0], coords[z_slice_dla, 1], s=0.0006, c=color_dla)
    ax[0].scatter([], [], c='black', label='All gas')
    ax[0].scatter([], [], c=color_dla, label='DLA selected')

    color_vr =  '#fae3d4'
    angle = np.linspace(0, 2*np.pi, 100)
    for i in range(len(halocoord[z_slice_haloes])):
        if halomass[z_slice_haloes][i] > 1e8/(1e10/lib.h):
            x = halocoord[z_slice_haloes][i, 0] + haloradius[z_slice_haloes][i]*np.cos(angle)
            y = halocoord[z_slice_haloes][i, 1] + haloradius[z_slice_haloes][i]*np.sin(angle)
            ax[0].plot(x, y, color=color_vr, lw=1.5)
    ax[0].set_xlim(centre[0]-xy_width, centre[0]+xy_width)
    ax[0].set_ylim(centre[1]-xy_width, centre[1]+xy_width)
    ax[0].legend(frameon=True, framealpha=0.5, fontsize=13)
    
            
    
    #### SLICES
    h_density = h_density[z_slice]
    temperature = temperature[z_slice]
    metallicity = metallicity[z_slice]
    gasmass = gasmass[z_slice]
    oxygen_density = oxygen_density[z_slice]
    iron_density = iron_density[z_slice]
    carbon_density = carbon_density[z_slice]
    silicon_density = silicon_density[z_slice]

    PlotSize=300
    Nplot=300
    centre = targethalo.data[snap]['coord']
    gas_coord_centred = coords[z_slice] - centre
    Edges1d = np.linspace(-PlotSize, PlotSize, Nplot+1, endpoint=True, dtype=np.float64)
    Grid1d = 0.5*(Edges1d[1:] + Edges1d[:-1])

    xgrid, ygrid, zgrid = np.meshgrid(Grid1d, Grid1d, Grid1d)
    Grid3D = np.array( [xgrid.reshape(Nplot**3), ygrid.reshape(Nplot**3), zgrid.reshape(Nplot**3)]).T
    _, cells = spatial.KDTree(np.array(gas_coord_centred[:,:])).query( Grid3D, k=1 )
    weight_slice = (gasmass[cells]).reshape((Nplot,Nplot,Nplot))

    density_slice = (h_density[cells]).reshape((Nplot,Nplot,Nplot))
    density_slice = np.sum(density_slice*weight_slice, axis=2)/np.sum(weight_slice, axis=2)

    metallicity_slice = (metallicity[cells]).reshape((Nplot,Nplot,Nplot))
    metallicity_slice = np.sum(metallicity_slice*weight_slice, axis=2)/np.sum(weight_slice, axis=2)

    temperature_slice = (temperature[cells]).reshape((Nplot,Nplot,Nplot))
    temperature_slice = np.sum(temperature_slice*weight_slice, axis=2)/np.sum(weight_slice, axis=2)

    vmin_dens = np.percentile(np.log10(density_slice),2) 
    vmax_dens = np.percentile(np.log10(density_slice),98)
    im1 = ax[1].imshow(np.log10(density_slice), origin='lower', extent=(-PlotSize, PlotSize, -PlotSize, PlotSize), cmap=cmc.lajolla, vmin=vmin_dens, vmax=vmax_dens)
    add_cbar(fig, im1, ax[1], label=r'log ($n_{\rm H}$ [cm$^{-3}$])', color='white')

    vmin_temp = np.percentile(np.log10(temperature_slice),2)
    vmax_temp = np.percentile(np.log10(temperature_slice),98)
    im2 = ax[2].imshow(np.log10(temperature_slice), origin='lower', extent=(-PlotSize, PlotSize, -PlotSize, PlotSize), cmap=cmc.lipari, vmin=vmin_temp, vmax=vmax_temp)
    add_cbar(fig, im2, ax[2], label=r'log (T [K])', color='white')  
    
    for j in range(len(ax)-1):
        for i in range(len(halocoord[z_slice_haloes])):
            if halomass[z_slice_haloes][i] > 1e8/(1e10/lib.h):
                x = halocoord[z_slice_haloes][i, 0] + haloradius[z_slice_haloes][i]*np.cos(angle) - centre[0]
                y = halocoord[z_slice_haloes][i, 1] + haloradius[z_slice_haloes][i]*np.sin(angle) - centre[1]
                if j==2 or j==5:
                    ax[j+1].plot(x, y, color='black', lw=1.5)
                else:
                    ax[j+1].plot(x, y, color=color_vr, lw=1.5)

    for j in range(len(ax)):
        ax[j].set_xticks([])
        ax[j].set_yticks([])
        if j >0:
            ax[j].set_xlim(-PlotSize, PlotSize)
            ax[j].set_ylim(-PlotSize, PlotSize)
        
    
    o_fe_slice = (oxygen_density[cells]/iron_density[cells]).reshape((Nplot,Nplot,Nplot))
    o_fe_slice = np.sum(o_fe_slice*weight_slice, axis=2)/np.sum(weight_slice, axis=2)
    o_fe_slice = np.log10(o_fe_slice) - mlib.O_Fe_solar
    if gal=='g39':
        vmin_o_fe = 0.05
        vmax_o_fe = 0.7
        cbarcol='black'
    elif gal=='g5760':
        vmin_o_fe = -0.5
        vmax_o_fe = 0.9
        cbarcol = 'azure'
    else:
        vmin_o_fe = np.percentile(o_fe_slice,2)
        vmax_o_fe = np.percentile(o_fe_slice,98)
    im3 = ax[3].imshow(o_fe_slice, origin='lower', extent=(-PlotSize, PlotSize, -PlotSize, PlotSize), cmap=cmc.lajolla, vmin=vmin_o_fe, vmax=vmax_o_fe)
    add_cbar(fig, im3, ax[3], label=r'[O/Fe]', color=cbarcol, label_color='black')

    c_fe_slice = (carbon_density[cells]/iron_density[cells]).reshape((Nplot,Nplot,Nplot))
    c_fe_slice = np.sum(c_fe_slice*weight_slice, axis=2)/np.sum(weight_slice, axis=2)
    c_fe_slice = np.log10(c_fe_slice) - mlib.C_Fe_solar
    if gal=='g39':
        vmin_c_fe = 0
        vmax_c_fe = 0.35
        cbarcol='black'
    elif gal=='g5760':
        vmin_c_fe = -0.35
        vmax_c_fe = 0.52
        cbarcol = 'azure'
    else:
        vmin_c_fe = np.percentile(c_fe_slice,2)
        vmax_c_fe = np.percentile(c_fe_slice,98)
    im4 = ax[4].imshow(c_fe_slice, origin='lower', extent=(-PlotSize, PlotSize, -PlotSize, PlotSize), cmap=cmc.lipari, vmin=vmin_c_fe, vmax=vmax_c_fe)
    
    add_cbar(fig, im4, ax[4], label=r'[C/Fe]', color=cbarcol, label_color='black')

    si_c_slice = (silicon_density[cells]/carbon_density[cells]).reshape((Nplot,Nplot,Nplot))
    si_c_slice = np.sum(si_c_slice*weight_slice, axis=2)/np.sum(weight_slice, axis=2)
    si_c_slice = np.log10(si_c_slice) - mlib.Si_C_solar
    if gal=='g39':
        vmin_si_c = 0
        vmax_si_c = 0.3
        cbarcol = 'black'
    elif gal=='g5760':
        vmin_si_c = 0.4
        vmax_si_c = 0.55
        cbarcol = 'black'
    else:
        vmin_si_c = np.percentile(si_c_slice,2)
        vmax_si_c = np.percentile(si_c_slice,98)
    im5 = ax[5].imshow(si_c_slice, origin='lower', extent=(-PlotSize, PlotSize, -PlotSize, PlotSize), cmap=cmc.navia, vmin=vmin_si_c, vmax=vmax_si_c)
    add_cbar(fig, im5, ax[5], label=r'[Si/C]', color=cbarcol)

    si_fe_slice = (silicon_density[cells]/iron_density[cells]).reshape((Nplot,Nplot,Nplot))
    si_fe_slice = np.sum(si_fe_slice*weight_slice, axis=2)/np.sum(weight_slice, axis=2)
    si_fe_slice = np.log10(si_fe_slice) - mlib.Si_Fe_solar
    if gal=='g39':
        vmin_si_fe = 0.11
        vmax_si_fe = 0.29
        cbarcol = 'black'
    elif gal=='g5760':
        vmin_si_fe = 0.27
        vmax_si_fe = 0.35
        cbarcol = 'black'
    else:
        vmin_si_fe = np.percentile(si_fe_slice,2)
        vmax_si_fe = np.percentile(si_fe_slice,98)
    im6 = ax[6].imshow(si_fe_slice, origin='lower', extent=(-PlotSize, PlotSize, -PlotSize, PlotSize), cmap=cmc.acton, vmin=vmin_si_fe, vmax=vmax_si_fe)
    add_cbar(fig, im6, ax[6], label=r'[Si/Fe]', color=cbarcol)

    si_o_slice = (silicon_density[cells]/oxygen_density[cells]).reshape((Nplot,Nplot,Nplot))
    si_o_slice = np.sum(si_o_slice*weight_slice, axis=2)/np.sum(weight_slice, axis=2)
    si_o_slice = np.log10(si_o_slice) - mlib.Si_O_solar
    if gal=='g39':
        vmin_si_o = -0.37
        vmax_si_o = 0.42
        cbarcol = 'black'
    elif gal=='g5760':
        vmin_si_o = 0.6
        vmax_si_o = 0.92
        cbarcol = 'black'
    else:
        vmin_si_o = np.percentile(si_o_slice,2)
        vmax_si_o = np.percentile(si_o_slice,98)
    im7 = ax[7].imshow(si_o_slice, origin='lower', extent=(-PlotSize, PlotSize, -PlotSize, PlotSize), cmap=cmc.lapaz, vmin=vmin_si_o, vmax=vmax_si_o)
    add_cbar(fig, im7, ax[7], label=r'[Si/O]', color=cbarcol)

    c_o_slice = (carbon_density[cells]/oxygen_density[cells]).reshape((Nplot,Nplot,Nplot))
    c_o_slice = np.sum(c_o_slice*weight_slice, axis=2)/np.sum(weight_slice, axis=2)
    c_o_slice = np.log10(c_o_slice) - mlib.C_O_solar
    if gal=='g39':
        vmin_c_o = -0.43
        vmax_c_o = 0.18
        cbarcol = 'black'
    elif gal=='g5760':
        vmin_c_o = 0.1
        vmax_c_o = 0.42
        cbarcol='black'
    else:
        vmin_c_o = np.percentile(c_o_slice,2)
        vmax_c_o = np.percentile(c_o_slice,98)
    im8 = ax[8].imshow(c_o_slice, origin='lower', extent=(-PlotSize, PlotSize, -PlotSize, PlotSize), cmap=cmc.devon, vmin=vmin_c_o, vmax=vmax_c_o)
    add_cbar(fig, im8, ax[8], label=r'[C/O]', color=cbarcol)
    
    fig.savefig('/home/gpruto/metal_ab/images/paper/dla_cuts_combinedmap_%s_z%d.png' %(gal, redshift), bbox_inches='tight', dpi=300)
    