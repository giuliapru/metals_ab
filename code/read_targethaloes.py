import numpy as np
import matplotlib.pyplot as plt
plt.style.use('/home/gpruto/CGM_galaxies/paper.style')
import h5py
import sys
import os
sys.path.append('/home/gpruto/CGM_ref_analysis/code')
import lib
sys.path.append('/home/gpruto/CGM_ref_analysis/code/metals')
import metals_lib as mlib
from tqdm.notebook import tqdm as progressbar
from haloes_class import TargetHalo
from mpl_toolkits.mplot3d import Axes3D
from scipy import spatial
from shapely.geometry import Point, Polygon
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
import cmcrameri.cm as cmc
import matplotlib.colors

snap = int(sys.argv[1]) #92, 68, 51
run = 'fiducial' 
cond_hr = 0.5

gal = ['g5229300', 'g2274036', 'g519761', 'g500531', 'g137030', 'g37591','g33206', 'g10304', 'g5760', 'g1163', 'g578', 'g205', 'g39']

for g in gal:  
    targethalo = TargetHalo(g, run)
    targethalo.read_haloes(snap, 0)
    halo_mass = targethalo.data[snap]['mass200']*1e10/lib.h
    coords, volume, redshift, gasmass, _, _, h_density, hi_density, carbon_density, oxygen_density, silicon_density, iron_density, temperature = targethalo.gas_properties(snap, cond_hr, 1., all=False, metals=True, temperature=True)
    

    _, star_initial_mass, _, star_birthday, star_mass = targethalo.stars_within_radius(snap, 1.)

    cond_10 = lib.time(star_birthday) > (lib.time(1/(1+targethalo.data[snap]['redshift'])) - 0.01)
    cond_100 = lib.time(star_birthday) > (lib.time(1/(1+targethalo.data[snap]['redshift'])) - 0.1)
    sfr_10 = np.sum(star_mass[cond_10]*1e10/lib.h)/1e7
    sfr_100 = np.sum(star_mass[cond_100]*1e10/lib.h)/1e8
    
    stellar_mass = np.sum(star_mass*1e10/lib.h)

    neutral_oxygen_density = oxygen_density*hi_density/h_density


    with h5py.File('/cephfs/gpruto/metals/target_haloes/z%d/%s.hdf5' % (int(round(redshift)), g), 'a') as f:
        #add to header
        f.attrs['halo_mass'] = halo_mass
        f.attrs['sfr_10'] = sfr_10
        f.attrs['sfr_100'] = sfr_100
        f.attrs['stellar_mass'] = stellar_mass
        f.attrs['redshift'] = redshift
        #add datasets
        f.create_dataset('h_density', data=h_density)
        f.create_dataset('hi_density', data=hi_density)
        f.create_dataset('carbon_density', data=carbon_density)
        f.create_dataset('oxygen_density', data=oxygen_density)
        f.create_dataset('silicon_density', data=silicon_density)
        f.create_dataset('iron_density', data=iron_density)
        f.create_dataset('temperature', data=temperature)
        f.create_dataset('volume', data=volume)
        f.create_dataset('mass', data=gasmass)