import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import healpy as hp
import os,sys
import numpy as np
import pyccl as ccl
import pylab as plt
import matplotlib.cm as cm
from scipy.special import erf
from scipy.stats import norm
from astropy.io import fits
from cobaya.run import run
from astropy.io import fits
from getdist import loadMCSamples
import cosmopower as cp


def rebincl(ell,cl, bb):
    #bb   = np.linspace(minell,maxell,Nbins+1)
    Nbins=len(bb)-1
    ll   = (bb[:-1]).astype(np.int_)
    uu   = (bb[1:]).astype(np.int_)
    ret  = np.zeros(Nbins)
    retl = np.zeros(Nbins)
    err  = np.zeros(Nbins)
    for i in range(0,Nbins):
        ret[i]  = np.mean(cl[ll[i]:uu[i]])
        retl[i] = np.mean(ell[ll[i]:uu[i]])
        err[i]  = np.std(cl[ll[i]:uu[i]])
    return ret


seed=1


# Fiducial cosmology (note: cov was accidently created using Omegac=0.315 not 0.3175)
cosmo = ccl.Cosmology(Omega_c=0.3175-0.049, Omega_b=0.049, h=0.677, sigma8=0.8, n_s=0.9624)
k_arr = np.geomspace(1E-4,1E1,256)
a_arr = np.linspace(0.1,1,128)


# Redshift distribution as defined in Porqueres
z_arr=np.linspace(0.1, 2.5, 500)
nz1=norm.pdf(np.linspace(0.1, 2.5, 500), loc=0.5, scale=0.12)
nz2=norm.pdf(np.linspace(0.1, 2.5, 500), loc=1.0, scale=0.12)
nz3=norm.pdf(np.linspace(0.1, 2.5, 500), loc=1.5, scale=0.12)
nz4=norm.pdf(np.linspace(0.1, 2.5, 500), loc=2.0, scale=0.12)


t_M1   = ccl.WeakLensingTracer(cosmo, dndz=(z_arr, nz1))
t_M2   = ccl.WeakLensingTracer(cosmo, dndz=(z_arr, nz2))
t_M3   = ccl.WeakLensingTracer(cosmo, dndz=(z_arr, nz3))
t_M4   = ccl.WeakLensingTracer(cosmo, dndz=(z_arr, nz4))


#hmd_200m = ccl.halos.MassDef200c
#cM = ccl.halos.ConcentrationDuffy08()
#nM = ccl.halos.MassFuncTinker08(cosmo,mass_def=hmd_200m)
#bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200m)
#hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200m)
#pM     = ccl.halos.HaloProfileNFW(mass_def=hmd_200m, concentration=cM, fourier_analytic=True)

mass_def = ccl.halos.massdef.MassDef200m()
cM       = ccl.halos.concentration.ConcentrationDuffy08(mass_def)
hmf      = ccl.halos.MassFuncTinker08(cosmo, mass_def=mass_def)
hbf      = ccl.halos.HaloBiasTinker10(cosmo, mass_def=mass_def)
hmc      = ccl.halos.HMCalculator(cosmo, hmf, hbf, mass_def,nlog10M=31,log10M_max=15.0)
prfK     = ccl.halos.HaloProfileNFW(cM)
pk_MMf   = ccl.halos.halomod_Pk2D(cosmo, hmc, prfK, lk_arr=np.log(k_arr), a_arr=a_arr)

l_arr = np.arange(513)
cl_11 = ccl.angular_cl(cosmo, t_M1, t_M1, l_arr, p_of_k_a=pk_MMf)
cl_12 = ccl.angular_cl(cosmo, t_M1, t_M2, l_arr, p_of_k_a=pk_MMf)
cl_13 = ccl.angular_cl(cosmo, t_M1, t_M3, l_arr, p_of_k_a=pk_MMf)
cl_14 = ccl.angular_cl(cosmo, t_M1, t_M4, l_arr, p_of_k_a=pk_MMf)
cl_22 = ccl.angular_cl(cosmo, t_M2, t_M2, l_arr, p_of_k_a=pk_MMf)
cl_23 = ccl.angular_cl(cosmo, t_M2, t_M3, l_arr, p_of_k_a=pk_MMf)
cl_24 = ccl.angular_cl(cosmo, t_M2, t_M4, l_arr, p_of_k_a=pk_MMf)
cl_33 = ccl.angular_cl(cosmo, t_M3, t_M3, l_arr, p_of_k_a=pk_MMf)
cl_34 = ccl.angular_cl(cosmo, t_M3, t_M4, l_arr, p_of_k_a=pk_MMf)
cl_44 = ccl.angular_cl(cosmo, t_M4, t_M4, l_arr, p_of_k_a=pk_MMf)


Lx = 20. * np.pi/180
Ly = 20. * np.pi/180
#  - Nx and Ny: the number of pixels in the x and y dimensions
Nx = 80
Ny = 80

l0_bins = np.arange(Nx/8) * 8 * np.pi/Lx
lf_bins = (np.arange(Nx/8)+1) * 8 * np.pi/Lx
bb  = l0_bins[0:6] #bb  = l0_bins[2:18]


# simulated dvec
bcl11 = rebincl(l_arr,cl_11,bb)
bcl12 = rebincl(l_arr,cl_12,bb)
bcl13 = rebincl(l_arr,cl_13,bb)
bcl14 = rebincl(l_arr,cl_14,bb)
bcl22 = rebincl(l_arr,cl_22,bb)
bcl23 = rebincl(l_arr,cl_23,bb)
bcl24 = rebincl(l_arr,cl_24,bb)
bcl33 = rebincl(l_arr,cl_33,bb)
bcl34 = rebincl(l_arr,cl_34,bb)
bcl44 = rebincl(l_arr,cl_44,bb)
bcl    = np.concatenate([bcl11,bcl22,bcl33,bcl44,bcl12,bcl23,bcl34,bcl13,bcl24,bcl14])

#cov
cl = np.zeros(((len(bb)-1)*10,1000))
d='/lcrc/project/SPT3G/users/ac.yomori/repo/sbi_lens/pspec2/'
for i in range(0,1000):
    f=np.load(d+'cls_seed%d.npz'%i)
    cl11=f['(0,0)'][0][0:5]
    cl22=f['(1,1)'][0][0:5]
    cl33=f['(2,2)'][0][0:5]
    cl44=f['(3,3)'][0][0:5]
    cl12=f['(0,1)'][0][0:5]
    cl23=f['(1,2)'][0][0:5]
    cl34=f['(2,3)'][0][0:5]
    cl13=f['(0,2)'][0][0:5]
    cl24=f['(1,3)'][0][0:5]
    cl14=f['(0,3)'][0][0:5]
    clall=np.concatenate([cl11,cl22,cl33,cl44,cl12,cl23,cl34,cl13,cl24,cl14])    
    cl[:,i]=clall

cov = np.cov(cl)


hartlap = (1000-2-len(bcl))/(1000-1)
icov    = np.linalg.inv(cov)*hartlap

emul = cp.cosmopower_PCAplusNN(restore=True,restore_filename='PP_cp_PCAplusNN.npz')

def dlike(omegac,sigma8):
    testp = {'omegac' : [ omegac ],
             'sigma8' : [ sigma8],
            }

    scl   = emul.predictions_np(testp)[0]
    
    X      = bcl-scl
    chi2   = X @ icov @ X
    return -0.5*chi2



pars  = {
          'omegac'    : {'prior': {'min': 0.2-0.049  , 'max': 0.7-0.049} , 'latex': r'\Omega_{\rm c}'},
          'sigma8'    : {'prior': {'min': 0.5  , 'max': 1.6} , 'latex': r'\sigma_{8}'}
        }

info =  {
          "params"    : pars,
          "likelihood": {'testlike': {"external": dlike } },
          "sampler"  : {"mcmc": {"max_samples": 100000, "Rminus1_stop": 0.05, "max_tries": 100000}},
          #"sampler"   : {"polychord": {"path":'/lcrc/project/SPT3G/users/ac.yomori/scratch/testcobaya3/PolyChordLite/',
          #                             "precision_criterion": 0.01,
          #                             'nlive': '10d',
          #                             'num_repeats': '3d'}},
          "output"    : 'chain_porqueres_mh_lmin0_lmax400_matchedres_emul.txt',
          "force"     : True,
          'feedback'  : 9999999
        }

print('-----------------------SAMPLING------------------------------')
updated_info, products = run(info)















