# -*- coding: utf-8 -*-

# performs a 2D simulation, assuming cylinder symmertry along Z

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io
import scipy as scipy
import Helmholtz as Helm

import time as time


# %load_ext autoreload

def save(filename, numpyarray):
    import scipy.io
    data = {}
    data['numpyarray'] = numpyarray
    scipy.io.savemat(filename, mdict=data)



# CUDA_DEVICE_VISIBLE=all my gpus

is_debug = False

mysize = (1024, 512)
myN = np.zeros(mysize)

SlabX = 260;
SlabW = 50;
SlabN = 1.52 + 0.0 * 1j;

# generate Sample
myN = Helm.insertSphere((myN.shape[0], myN.shape[1], 1), obj_dim=0.1, obj_type=0, diameter=5, dn=SlabN)
plt.imshow(np.squeeze(np.real(myN)))

k0 = 0.25 / abs(SlabN);
myN, _ = Helm.insertPerfectAbsorber(myN, 0, 100, -1, k0);
myN, _ = Helm.insertPerfectAbsorber(myN, myN.shape[0] - 100, 100, 1, k0);

if is_debug:
    plt.imshow(np.squeeze(np.abs(myN[:, :, :])))
    plt.show()
    plt.imshow(np.squeeze(np.angle(myN[:, :, :])))
    plt.show()
save('myN', myN)

kx = 0.3;
myWidth = 60;
mySrc = Helm.insertSrc(mysize, myWidth, myOff=(101, 80), kx=kx);
if is_debug:
    plt.imshow((np.angle(mySrc)))
    plt.show()
    plt.imshow((np.abs(mySrc)))
    plt.show()
    # Currently only running in 2D


# Instantiate the Helmholtzsolver
MyHelm = Helm.HelmholtzSolver(myN, mySrc, myeps=None, k0=k0, startPsi=None, showupdate=10)

# Compute the model inside the convergent born series 
MyHelm.computeModel()

# Initialize all operands
MyHelm.compileGraph()

#%% Do n iterations to let the series converge
for i in range(10):

    # Compute n steps
    start_time = time.time()
    MyHelm.step(nsteps = 100)    
    print('Preparation took '+str(time.time()-start_time)+' s')
    
    # Evaluate/compute result from one step
    start_time = time.time()
    MyHelm.evalStep()
    print('Calculation took '+str(time.time()-start_time)+' s')
    
    # Display result 
    # Display result 
    plt.subplot(1,2,1), plt.title('Magnitude of PSI')
    plt.imshow(np.abs(MyHelm.psi_result)), plt.colorbar()
    plt.subplot(1,2,2), plt.title('Phase of PSI')
    plt.imshow(np.angle(MyHelm.psi_result)), plt.colorbar()
    plt.show()

#save('np_res', MyHelm.psi_result)
