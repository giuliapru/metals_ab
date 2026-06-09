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

fig, ax = plt.subplots(2,2, figsize=(10,10))
ax = ax.flatten()

mlib.plot_models(mlib.CFe0_WW, mlib.OFe0_WW, ax[0], 'gray')
mlib.plot_models(mlib.CFe1_WW, mlib.SiC1_WW, ax[1], 'gray')
mlib.plot_models(mlib.OFe2_WW, mlib.SiFe2_WW, ax[2], 'gray')
mlib.plot_models(mlib.CO3_WW, mlib.SiO3_WW, ax[3], 'gray')


mlib.plot_Sodini(mlib.CFe_Sodini, mlib.OFe_Sodini, mlib.CFe_err_Sodini, mlib.OFe_err_Sodini, mlib.CFe_limit_type, mlib.OFe_limit_type, ax[0], color='red', special=True)
mlib.plot_Sodini(mlib.CFe_Becker, mlib.OFe_Becker, mlib.CFe_Becker_err, mlib.OFe_Becker_err, mlib.CFe_Becker_limit_type, mlib.OFe_Becker_limit_type, ax[0], color='blue')
mlib.plot_Sodini(mlib.CFe_Poudel_2018, mlib.OFe_Poudel_2018, mlib.CFe_Poudel_2018_err, mlib.OFe_Poudel_2018_err, mlib.CFe_Poudel_2018_limit_type, mlib.OFe_Poudel_2018_limit_type, ax[0], color='green')
mlib.plot_Sodini(mlib.CFe_Poudel[:2], mlib.OFe_Poudel[:2], mlib.CFe_Poudel_err[:2], mlib.OFe_Poudel_err[:2], mlib.CFe_Poudel_limit_type[:2], mlib.OFe_Poudel_limit_type[:2], ax[0], color='green')
ax[0].set_xlabel(r'[C/Fe]')
ax[0].set_ylabel(r'[O/Fe]')

mlib.plot_Sodini(mlib.CFe_Sodini, mlib.SiC_Sodini, mlib.CFe_err_Sodini, mlib.SiC_err_Sodini, mlib.CFe_limit_type, mlib.SiC_limit_type, ax[1], color = 'red', special=True)
mlib.plot_Sodini(mlib.CFe_Becker, mlib.SiC_Becker, mlib.CFe_Becker_err, mlib.SiC_Becker_err, mlib.CFe_Becker_limit_type, mlib.SiC_Becker_limit_type, ax[1], color='blue')
mlib.plot_Sodini(mlib.CFe_Poudel_2018, mlib.SiC_Poudel_2018, mlib.CFe_Poudel_2018_err, mlib.SiC_Poudel_2018_err, mlib.CFe_Poudel_2018_limit_type, mlib.SiC_Poudel_2018_limit_type, ax[1], color='green')
mlib.plot_Sodini(mlib.CFe_Poudel[:2], mlib.SiC_Poudel[:2], mlib.CFe_Poudel_err[:2], mlib.SiC_Poudel_err[:2], mlib.CFe_Poudel_limit_type[:2], mlib.SiC_Poudel_limit_type[:2], ax[1], color='green')
ax[1].set_xlabel(r'[C/Fe]')
ax[1].set_ylabel(r'[Si/C]')

mlib.plot_Sodini(mlib.OFe_Sodini, mlib.SiFe_Sodini, mlib.OFe_err_Sodini, mlib.SiFe_err_Sodini, mlib.OFe_limit_type, mlib.SiFe_limit_type, ax[2], color='red', special=True)
mlib.plot_Sodini(mlib.OFe_Becker, mlib.SiFe_Becker, mlib.OFe_Becker_err, mlib.SiFe_Becker_err, mlib.OFe_Becker_limit_type, mlib.SiFe_Becker_limit_type, ax[2], color='blue')
mlib.plot_Sodini(mlib.OFe_Poudel_2018, mlib.SiFe_Poudel_2018, mlib.OFe_Poudel_2018_err, mlib.SiFe_Poudel_2018_err, mlib.OFe_Poudel_2018_limit_type, mlib.SiFe_Poudel_2018_limit_type, ax[2], color='green')
mlib.plot_Sodini(mlib.OFe_Poudel[:2], mlib.SiFe_Poudel[:2], mlib.OFe_Poudel_err[:2], mlib.SiFe_Poudel_err[:2], mlib.OFe_Poudel_limit_type[:2], mlib.SiFe_Poudel_limit_type[:2], ax[2], color='green')
ax[2].set_xlabel(r'[O/Fe]')
ax[2].set_ylabel(r'[Si/Fe]')

mlib.plot_Sodini(mlib.CO_Sodini, mlib.SiO_Sodini, mlib.CO_err_Sodini, mlib.SiO_err_Sodini, mlib.CO_limit_type, mlib.SiO_limit_type, ax[3], color='red', special=True)
mlib.plot_Sodini(mlib.CO_Becker, mlib.SiO_Becker, mlib.CO_Becker_err, mlib.SiO_Becker_err, mlib.CO_Becker_limit_type, mlib.SiO_Becker_limit_type, ax[3], color='blue')
mlib.plot_Sodini(mlib.CO_Poudel_2018, mlib.SiO_Poudel_2018, mlib.CO_Poudel_2018_err, mlib.SiO_Poudel_2018_err, mlib.CO_Poudel_2018_limit_type, mlib.SiO_Poudel_2018_limit_type, ax[3], color='green')
mlib.plot_Sodini(mlib.CO_Poudel[:2], mlib.SiO_Poudel[:2], mlib.CO_Poudel_err[:2], mlib.SiO_Poudel_err[:2], mlib.CO_Poudel_limit_type[:2], mlib.SiO_Poudel_limit_type[:2], ax[3], color='green')
ax[3].set_xlabel(r'[C/O]')
ax[3].set_ylabel(r'[Si/O]')

ax[0].set_xlim(-1., 1.5)
ax[0].set_ylim(-1., 2)
ax[1].set_xlim(-1., 1.5)
ax[1].set_ylim(-1., 1.5)
ax[2].set_xlim(-0.5, 2.)
ax[2].set_ylim(-0.5, 2)
ax[3].set_xlim(-1.5, 1.5)
ax[3].set_ylim(-2, 1)


fig.savefig('/home/gpruto/CGM_ref_analysis/images/metals/test_model_plot.png', dpi=300)






