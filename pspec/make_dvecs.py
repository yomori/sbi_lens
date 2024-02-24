import healpy as hp
import os,sys
import numpy as np
import pyccl as ccl
import pylab as plt
import matplotlib.cm as cm
from scipy.special import erf
from scipy.stats import norm
from astropy.io import fits

seed=int(sys.argv[1])


# Fiducial cosmology
cosmo = ccl.Cosmology(Omega_c=0.315-0.049, Omega_b=0.049, h=0.677, sigma8=0.8, n_s=0.9624)
k_arr = np.geomspace(1E-4,1E1,256)
a_arr = np.linspace(0.1,1,128)



#z_arr  = np.linspace(0,0.5,256)
#nz_arr = np.exp(-((z_arr - 0.25) / 0.05)**2)

z_arr=np.linspace(0.1, 2.5, 500)
nz1=norm.pdf(np.linspace(0.1, 2.5, 500), loc=0.5, scale=0.12) 
nz2=norm.pdf(np.linspace(0.1, 2.5, 500), loc=1.0, scale=0.12) 
nz3=norm.pdf(np.linspace(0.1, 2.5, 500), loc=1.5, scale=0.12) 
nz4=norm.pdf(np.linspace(0.1, 2.5, 500), loc=2.0, scale=0.12) 


t_M1   = ccl.WeakLensingTracer(cosmo, dndz=(z_arr, nz1))
t_M2   = ccl.WeakLensingTracer(cosmo, dndz=(z_arr, nz2))
t_M3   = ccl.WeakLensingTracer(cosmo, dndz=(z_arr, nz3))
t_M4   = ccl.WeakLensingTracer(cosmo, dndz=(z_arr, nz4))


hmd_200m = ccl.halos.MassDef200m

cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200m)
nM = ccl.halos.MassFuncTinker08(mass_def=hmd_200m)
bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200m)
hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200m)
pM     = ccl.halos.HaloProfileNFW(mass_def=hmd_200m, concentration=cM, fourier_analytic=True)
pk_MMf = ccl.halos.halomod_Pk2D(cosmo, hmc, pM, lk_arr=np.log(k_arr), a_arr=a_arr)

l_arr = np.arange(2048)
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

sige = 0.3 #per-component
nbar = np.array([7.0 ,8.5, 7.5, 7.0])
#nbar1,nbar2,nbar3,nbar4= 0.1 ,0.1, 0.1, 0.1

nl= sige**2/(nbar*3437.75**2)#**-1


alms=hp.synalm(np.c_[cl_11,cl_22,cl_33,cl_44,cl_12,cl_23,cl_34,cl_13,cl_24,cl_14].T,new=True)

#nlm1=hp.synalm(nl_11*np.ones_like(l_arr))
#nlm2=hp.synalm(nl_22*np.ones_like(l_arr))
#nlm3=hp.synalm(nl_33*np.ones_like(l_arr))
#nlm4=hp.synalm(nl_44*np.ones_like(l_arr))
#n1=hp.alm2map(nlm1,2048)
#n2=hp.alm2map(nlm2,2048)
#n3=hp.alm2map(nlm3,2048)
#n4=hp.alm2map(nlm4,2048)

m1=hp.alm2map(alms[0],2048)
m2=hp.alm2map(alms[1],2048)
m3=hp.alm2map(alms[2],2048)
m4=hp.alm2map(alms[3],2048)


#m1=m1#+n1
#m2=m2#+n2
#m3=m3#+n3
#m4=m4#+n4


import reproject

def set_header(ra,dec,span,size=500):
    #Sets the header of output projection
    #span = angular dimensions project
    #size = size of the output image
    res = span/(size+0.0)*0.0166667
    return hdr

def h2f(hmap,target_header,coord_in='C'):
    #project healpix -> flatsky
    pr,footprint = reproject.reproject_from_healpix(
    (hmap, coord_in), target_header, shape_out=(500,500),
    order='nearest-neighbor', nested=False)
    return pr

hdr = fits.Header()
hdr.set('NAXIS'   , 2)
hdr.set('NAXIS1'  , 800)
hdr.set('NAXIS2'  , 800)
hdr.set('CTYPE1'  , 'RA---ZEA')
hdr.set('CRPIX1'  , 800/2.0)
hdr.set('CRVAL1'  , 0.)
hdr.set('CDELT1'  , -0.0166667*1.2)
hdr.set('CUNIT1'  , 'deg')
hdr.set('CTYPE2'  , 'DEC--ZEA')
hdr.set('CRPIX2'  , 800/2.0)
hdr.set('CRVAL2'  , 0 )
hdr.set('CDELT2'  , 0.0166667*1.2)
hdr.set('CUNIT2'  , 'deg')
hdr.set('COORDSYS','icrs')

mpt={}
mpt[0]=h2f(m1,hdr,coord_in='C')
mpt[1]=h2f(m2,hdr,coord_in='C')
mpt[2]=h2f(m3,hdr,coord_in='C')
mpt[3]=h2f(m4,hdr,coord_in='C')


import pymaster as nmt

Lx = 18. * np.pi/180
Ly = 18. * np.pi/180
#  - Nx and Ny: the number of pixels in the x and y dimensions
Nx = 800
Ny = 800

d={}
for i in range(0,4):
    sig = sige/np.sqrt(nbar[i]*(18*60/Nx)**2)
    d[i] =np.random.normal(mpt[i],sig)

mask=np.ones((800,800))
mask[:28,:]=0
mask[-28:,:]=0
mask[:,:28]=0
mask[:,-28:]=0

from scipy.ndimage import gaussian_filter as gf
mask = gf(nmt.mask_apodization_flat(mask, Lx, Ly, aposize=0.77, apotype="C1"),20)

f  = {}
f[0] = nmt.NmtFieldFlat(Lx, Ly, mask, [d[0]])
f[1] = nmt.NmtFieldFlat(Lx, Ly, mask, [d[1]])
f[2] = nmt.NmtFieldFlat(Lx, Ly, mask, [d[2]])
f[3] = nmt.NmtFieldFlat(Lx, Ly, mask, [d[3]])


l0_bins = np.arange(Nx/8) * 8 * np.pi/Lx
lf_bins = (np.arange(Nx/8)+1) * 8 * np.pi/Lx
b = nmt.NmtBinFlat(l0_bins, lf_bins)
# The effective sampling rate for these bandpowers can be obtained calling:
ells_uncoupled = b.get_effective_ells()
cls={}
for i in range(0,4):
    for j in range(i,4):
        print('computing',i,j)
        w00 = nmt.NmtWorkspaceFlat()
        w00.compute_coupling_matrix(f[i], f[j], b)
        cl00_coupled   = nmt.compute_coupled_cell_flat(f[i], f[j], b)
        cl00_uncoupled = w00.decouple_cell(cl00_coupled)
        cls['(%d,%d)'%(i,j)] = cl00_uncoupled

np.savez('cls_seed%d.npz'%seed,**cls)
