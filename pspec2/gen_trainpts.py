import os
import sys
import numpy as np
from pathlib import Path
from scipy.stats import qmc
import datetime,argparse

nparams  = 2     # Number of parameters to vary
nsamples = 200    # Number of samples to take

file_out = 'params_train.txt'
sampler = qmc.LatinHypercube(d=nparams, optimization="random-cd")
sample  = sampler.random(n=nsamples)
qmc.discrepancy(sample)

range_omegac = [ 0.2 , 0.3175-0.049 , 0.7]
range_sigma8 = [ 0.5 , 0.8   , 1.6]

d = qmc.scale(sample, [range_omegac[0],range_sigma8[0]],
                      [range_omegac[2],range_sigma8[2]])

d = np.insert(d,0,[ range_omegac[1],range_sigma8[1]], axis=0)

np.savetxt(file_out,d,
           header='omegac[%.3f,%.3f] sigma8[%.3f,%.3f]'%(
                                                          range_omegac[0] ,  range_omegac[2],
                                                          range_sigma8[0] ,  range_sigma8[2],
                                                         )
          )

###############################################################################################
file_out = 'params_test.txt'
sampler  = qmc.LatinHypercube(d=nparams, optimization="random-cd")
sample   = sampler.random(n=nsamples)
qmc.discrepancy(sample)

d = qmc.scale(sample, [range_omegac[0],range_sigma8[0]],
                      [range_omegac[2],range_sigma8[2]])

d = np.insert(d,0,[ range_omegac[1],range_sigma8[1]], axis=0)

np.savetxt(file_out,d,
           header='omegac[%.3f,%.3f] sigma8[%.3f,%.3f]'%(
                                                          range_omegac[0] ,  range_omegac[2],
                                                          range_sigma8[0] ,  range_sigma8[2],
                                                         )
          )

