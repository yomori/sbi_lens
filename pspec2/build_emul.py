import numpy as np
import healpy as hp
import datetime
#import tensorflow as tf
#from cosmopower import cosmopower_PCA
#from cosmopower import cosmopower_PCAplusNN
import os,sys

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
    return retl,ret
        
# Bins
bine  = np.logspace(np.log10(100),np.log10(4000),21)
nbins = len(bine)-1

# Load paramaters and give each column names
file_train = 'params_train.txt'
pars       = np.loadtxt(file_train)
npars      = pars.shape[0]
par_names  = ['omegac','sigma8']

print('Loaded training parameter file: %s'%file_train)
print('Number of pars:', npars)
print('Column names  :', par_names)

# Load full debiased spectra == clkk-RDN0-N1
theory = np.load('theory_train.npy')

# Apply binning and store in array 
modes    = np.arange(nbins)             # number of modes
spectra  = np.zeros((npars-1,len(modes))) # name has to be spectra

for p in range(1,pars.shape[0]-1):
    l      = np.arange(4001)
    spectra[p-1,:] = theory[:,p-1]

pdict     = {par_names[i]: pars[i] for i in range(len(par_names))}

spec_dict = {'modes': modes, 'features': spectra}

# Save npz files for parameters and data vector
np.savez('params', **pdict)
np.savez('dvec'  , **spec_dict)


# Building the final estimator
np.random.seed(2)
tf.random.set_seed(3)


#location of parameter files of the form params.npz
parameters_filenames = ['params']
features_filenames   = ['dvec']

bins   = rl

cp_pca = cosmopower_PCA(parameters = par_names,
                        modes      = bins,
                        n_pcas     = 15,
                        parameters_filenames = parameters_filenames,
                        features_filenames   = features_filenames,
                        verbose    = True
                        )

cp_pca.transform_and_stack_training_data()

cp_pca_nn = cosmopower_PCAplusNN(cp_pca   = cp_pca,
                                 n_hidden = [128,128],
                                 verbose  = True,    
                                )

with tf.device('/device:GPU:0'): # ensures we are running on a GPU
    cp_pca_nn.train(filename_saved_model = 'PP_cp_PCAplusNN.npz',
                    validation_split     = 0.2,
                    learning_rates       = [1e-2,1e-3,1e-5,1e-6,1e-7,1e-8],
                    batch_sizes          = [4,4,4,4,4,4], # lower the better
                    patience_values      = [100,100,100,100,100,100],
                    max_epochs           = [2000,2000,4000,3000,3000,3000],
                    gradient_accumulation_steps = [1,1,1,1,1,1],
                    )