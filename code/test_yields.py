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

run = 'fiducial' 
cond_hr = 0.5
hist = True
cold_gas_only = str(sys.argv[2]) == 'True' #if True, only plot gas with T < 10^4 K, otherwise plot all gas
read_from_files = True #if true it reads data of gas particles from files in cephfs - it speeds up the process
snap = int(sys.argv[1]) #92, 68, 51
if snap == 92:
    red = 5
elif snap == 68:
    red = 6
elif snap == 51:
    red = 7
elif snap == 39:
    red = 8

silicon_scalef = 1



fig, ax = plt.subplots(1, 3, figsize=(17, 4))

for aa in ax:
    aa.set_xlim(-4.4, 0)
    aa.set_xlabel(r'[Fe/H]')

mlib.plot_Sodini(mlib.FeH_Sodini, mlib.SiFe_Sodini,  mlib.FeH_err_Sodini, mlib.SiFe_err_Sodini, mlib.FeH_err_limit_type, mlib.SiFe_limit_type, ax[0], color='red', special=True, label='Sodini+24')
mlib.plot_Sodini(mlib.FeH_Sodini, mlib.OFe_Sodini, mlib.FeH_err_Sodini, mlib.OFe_err_Sodini, mlib.FeH_err_limit_type, mlib.OFe_limit_type, ax[1], color='red', special=True)
mlib.plot_Sodini(mlib.FeH_Sodini, mlib.CFe_Sodini, mlib.FeH_err_Sodini, mlib.CFe_err_Sodini, mlib.FeH_err_limit_type, mlib.CFe_limit_type, ax[2], color='red', special=True)


FeH_kob06 = [-3.9480940794809407, -3.3122465531224656, -2.62124898621249, -2.0665044606650445, -1.557177615571776, -1.002433090024331, -0.5904298459042985, -0.1719383617193837]
SiFe_kob06 = [0.7950310559006211, 0.7329192546583849, 0.6428571428571428, 0.6055900621118011, 0.5838509316770186, 0.5496894409937887, 0.30434782608695654, 0.13664596273291907]

FeH_kob06_1 = [-3.9243336199484093, -3.4256233877901976, -2.999140154772141, -2.4797936371453138, -1.9604471195184865, -1.5511607910576095, -1.0146173688736027, -0.6981943250214959, -0.34049871023215816, 0.04815133276010286]
CFe_kob06_1 = [0.024752475247524774, 0.013613861386138515, -0.04207920792079212, -0.14603960396039606, -0.2240099009900991, -0.28712871287128716, -0.20915841584158423, -0.3613861386138614, -0.47648514851485146, -0.573019801980198]

FeH_kob06_2 = [-4.349588347055098, -3.946801773274224, -3.247625079164028, -2.654844838505383, -2.14059531348955, -1.6339455351488281, -1.2463584547181759, -0.9854338188727043, -0.6941101963267893, -0.4914502849905005, -0.21025965801139934, 0.12412919569347736]
OFe_kob06_2 = [0.31170108161258625, 0.32350049164208494, 0.177974434611603, 0.06391347099311728, 0.000983284169124854, -0.022615535889872085, -0.10521140609636159, -0.16027531956735475, -0.5260570304818093, -0.7581120943952802, -1.0019665683382497, -1.2576204523107177]

FeH_Portinari98 = [-2.940754039497307, -2.4775583482944343, -2.073608617594255, -1.6696588868940754, -1.3303411131059246, -0.9317773788150809, -0.4739676840215439, -0.03231597845601408]
SiFe_Portinari98 = [0.6132437619961613, 0.5287907869481765, 0.4750479846449136, 0.44049904030710174, 0.39827255278310936, 0.33685220729366605, 0.24472168905950098, 0.17562380038387715]

FeH_Portinari98_2 = [-2.9731182795698925, -2.553763440860215, -2.1666666666666665, -1.8387096774193548, -1.4139784946236558, -1.086021505376344, -0.7204301075268815, -0.36559139784946204, -0.06451612903225801]
OFe_Portinari98_2 = [0.6560693641618498, 0.5289017341040463, 0.44412331406551053, 0.39788053949903657, 0.3362235067437379, 0.27071290944123316, 0.15895953757225434, 0.03179190751445082, -0.1377649325626204]

FeH_Portinari98_1= [-2.941696113074205, -2.5547703180212014, -2.1996466431095407, -1.7173144876325088, -1.293286219081272, -0.7791519434628973, -0.49293286219081267, -0.31802120141342716, -0.16961130742049457, -0.03180212014134254]
CFe_Portinari98_1 = [0.28787878787878785, 0.18939393939393945, 0.13636363636363635, 0.10227272727272729, 0.06060606060606066, 0.11363636363636365, 0.28409090909090906, 0.3030303030303031, 0.21969696969696972, 0.12121212121212122]


ax[0].plot(FeH_kob06, SiFe_kob06, color='black', label='Kobayashi+06')
ax[0].plot(FeH_Portinari98, SiFe_Portinari98, color='tab:orange', label='Portinari+98', linestyle='dashed')
ax[1].plot(FeH_kob06_1, CFe_kob06_1, color='black', label='Kobayashi+06')
ax[2].plot(FeH_kob06_2, OFe_kob06_2, color='black', label='Kobayashi+06')
ax[1].plot(FeH_Portinari98_1, CFe_Portinari98_1, color='tab:orange', label='Portinari+98', linestyle='dashed')
ax[2].plot(FeH_Portinari98_2, OFe_Portinari98_2, color='tab:orange', label='Portinari+98', linestyle='dashed')


ax[0].legend()
ax[0].set_ylim(-1.4, 1.6)
ax[1].set_ylim(-1.4, 2)
ax[2].set_ylim(-3.4, 1.8)

ax[0].set_ylabel(r'[Si/Fe]')
ax[1].set_ylabel(r'[C/Fe]')
ax[2].set_ylabel(r'[O/Fe]')


gal = ['g5229300', 'g2274036', 'g519761', 'g500531', 'g137030', 'g37591','g33206', 'g10304', 'g5760', 'g1163', 'g578', 'g205', 'g39']
#colours = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black']


si_fe_allg = []
fe_h_allg = []
fe_hi_allg = []
o_fe_allg = []
c_fe_allg = []

norm = matplotlib.colors.Normalize(vmin=0, vmax=len(gal)-1)
colours=[]

for i in range(len(gal)):
    colours.append(cmc.lipari(norm(i)))

for g in gal: 
    #if you need to read from the simulation outputs
    if read_from_files==False:
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

    # if you can use files with ptcs in the CGM of target haloes already selected  
    else:
        infile = '/cephfs/gpruto/metals/target_haloes/z%d/%s.hdf5' %(red, g)
        with h5py.File(infile, 'r') as f:
            h_density = f['h_density'][:]
            hi_density = f['hi_density'][:]
            carbon_density = f['carbon_density'][:]
            oxygen_density = f['oxygen_density'][:]
            silicon_density = f['silicon_density'][:]
            iron_density = f['iron_density'][:]
            temperature = f['temperature'][:]
            volume = f['volume'][:]
            gasmass = f['mass'][:]

            halo_mass = f.attrs['halo_mass']
            stellar_mass = f.attrs['stellar_mass']
            sfr_10 = f.attrs['sfr_10']
            sfr_100 = f.attrs['sfr_100']

    if cold_gas_only:
            condition_cold = temperature < 1e4
            print('Number of ptcs within the virial radius: %.2f' % len(temperature), ' checking condition cold is of the same dimension: %.2f' %len(condition_cold))
            print('In galaxy %s, the number of cold gas particles is %.2f' % (g, len(condition_cold[condition_cold==True])))
            h_density = h_density[condition_cold]
            hi_density = hi_density[condition_cold]
            carbon_density = carbon_density[condition_cold]
            oxygen_density = oxygen_density[condition_cold]
            silicon_density = silicon_density[condition_cold]
            iron_density = iron_density[condition_cold]
            temperature = temperature[condition_cold]
            volume = volume[condition_cold]
            gasmass = gasmass[condition_cold]
            print('In galaxy %s, the number of cold gas particles after applying the condition is %d' % (g, len(temperature)))

    silicon_density *= silicon_scalef
    neutral_oxygen_density = oxygen_density*hi_density/h_density

    si_fe = np.log10(silicon_density/iron_density) - mlib.Si_Fe_solar
    fe_h = np.log10(iron_density/h_density) - mlib.Fe_sun
    fe_hi = np.log10(iron_density/hi_density) - mlib.Fe_sun
    o_fe = np.log10(oxygen_density/iron_density) - mlib.O_Fe_solar
    c_fe = np.log10(carbon_density/iron_density) - mlib.C_Fe_solar
    
    si_fe_allg.extend(np.array(si_fe).ravel().tolist())
    fe_h_allg.extend(np.array(fe_h).ravel().tolist())
    fe_hi_allg.extend(np.array(fe_hi).ravel().tolist())
    o_fe_allg.extend(np.array(o_fe).ravel().tolist())
    c_fe_allg.extend(np.array(c_fe).ravel().tolist())

if silicon_scalef!=1:
    mlib.hist_2d(np.array(fe_h_allg), np.array(si_fe_allg), ax[0], color='Blues')
else:
    mlib.hist_2d(np.array(fe_h_allg), np.array(si_fe_allg), ax[0])
mlib.hist_2d(np.array(fe_h_allg), np.array(c_fe_allg), ax[1])
mlib.hist_2d(np.array(fe_h_allg), np.array(o_fe_allg), ax[2])

infile = '/cephfs/gpruto/CGM_ref/arepo/data/Arepo_GFM_Tables/Yields/SNII.hdf5'
with h5py.File(infile,'r') as f:
    n_masses = int(f['Number_of_masses'][()])    # scalar -> python int
    print('Number_of_masses =', n_masses)
    n_metallicities = int(f['Number_of_metallicities'][()])
    print('Number_of_metallicities =', n_metallicities)
    n_species = int(f['Number_of_species'][()])
    print('Number_of_species =', n_species)
    species_names = f['Species_names'][:]
    print('Species_names =', species_names)
    yield_names = f['Yield_names'][:]
    print('Yield_names =', yield_names)
    print('Just to check... ', f['Yields']['Z_0.05']['Yield'])

    ejected_mass = f['Yields']['Z_0.05']['Ejected_mass'][:]
    print('Dimension of ejected mass ', ejected_mass.shape)

    # common checks:
    if 'Masses' in f:
        mass = f['Masses'][:]
        print('Masses =', mass)
    # check yields shape if present
    if 'Yields' in f:
        y = f['Yields']
        print(list(y.keys()))
        print(f['Yields']['Z_0.02'].keys())
        
        #print('sum over species ', np.sum(f['Yields']['Z_0.05']['Yield'][:,0]))
        #print(f['Yields']['Z_0.02']['Total_Metals'][:]) #*f['Yields']['Z_0.02']['Ejected_mass'][:])
        print(f['Yields']['Z_0.02']['Ejected_mass'][:])
        print(f['Yields']['Z_0.02']['Yield'][7, :]+ 6.7e-4*f['Yields']['Z_0.02']['Ejected_mass'][:])#*f['Yields']['Z_0.02']['Ejected_mass'][:])
        #print('total mass of silicon ejected ', (f['Yields']['Z_0.02']['Yield'][7,:]*f['Yields']['Z_0.02']['Ejected_mass'][:]))
        #obj = y[name]
        #print(obj.keys())
        # typical shapes: (n_z, n_mass, n_elem) or (n_mass, n_elem)

#<KeysViewHDF5 ['Masses', 'Metallicities', 'Number_of_masses', 'Number_of_metallicities', 'Number_of_species', 'Reference', 'Species_names', 'Yield_names', 'Yields']>

if silicon_scalef!=1:
    plt.savefig('/home/gpruto/CGM_ref_analysis/images/metals/test_yields__z%d_six%.1f.png' %(red, silicon_scalef), bbox_inches='tight', dpi=300)
else:
    plt.savefig('/home/gpruto/CGM_ref_analysis/images/metals/test_yields_z%d.png' %red, bbox_inches='tight', dpi=300)
