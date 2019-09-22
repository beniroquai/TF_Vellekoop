# -*- coding: utf-8 -*-

# performs a 2D simulation, assuming cylinder symmertry along Z


import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import Helmholtz as Helm
import time


# %load_ext autoreload
# %reload_ext autoreload


def save(filename, numpyarray):
    import scipy.io
    data = {}
    data['numpyarray'] = numpyarray
    scipy.io.savemat(filename, mdict=data)


# CUDA_DEVICE_VISIBLE=all my gpus

is_debug = False

if (1):
    #%% calculate the object 
    mysize = (100, 64, 64) # z,x,y
    dips_z, disp_x, disp_y = mysize[0]//2, mysize[1]//2, mysize[2]//2
    Boundary = 15
    
    # define object parameters
    myN = np.ones(mysize)
    n_obj = 1.52 + 0.0 * 1j;
    n_embb = 1.1
    lambda_0 = 4.#.65
    dn = np.abs(n_obj)-np.abs(n_embb)
    
    # generate Sample
    myN = Helm.insertSphere((myN.shape[0], myN.shape[1], myN.shape[2]), obj_dim=0.1, obj_type=0, diameter=1, dn=dn, n_0 = n_embb)

    k0 = 0.25 / abs(n_obj);
    myN, _ = Helm.insertPerfectAbsorber(myN, 0, Boundary, -1, k0);
    myN, _ = Helm.insertPerfectAbsorber(myN, myN.shape[0] - Boundary, Boundary, 1, k0);

    plt.imshow(np.squeeze(np.real(myN[:,:,np.int(np.floor(mysize[2]/2))]))), plt.colorbar(), plt.show()
    
    if is_debug:
        plt.imshow(np.squeeze(np.abs(myN[:, :, :])))
        plt.show()
        plt.imshow(np.squeeze(np.angle(myN[:, :, :])))
        plt.show()
    save('myN', myN)

    #%% compute the source
    kx = 0.0 # angle in X direction
    ky = 0. # angle in Y direction
    myWidth = 30;
    mySrc = Helm.insertSrc(mysize, myWidth, myOff=(Boundary+1, 0, 0), kx=kx,ky=ky)
    plt.imshow(np.squeeze(np.real(mySrc[:,:,np.int(np.floor(mysize[2]/2))]))), plt.colorbar(), plt.show()
    
    if is_debug:
        plt.imshow((np.angle(mySrc)))
        plt.show()
        plt.imshow((np.abs(mySrc)))
        plt.show()
        # Currently only running in 2D


# Instantiate the Helmholtzsolver
MyHelm = Helm.HelmholtzSolver(myN, myN0, dn, mySrc, myeps=None, lambda_0=lambda_0)

# Compute the model inside the convergent born series 
MyHelm.computeModel()

# Initialize all operands
MyHelm.compileGraph()

# Compute n steps
start_time = time.time()
MyHelm.step(nsteps = 20)    
print('Preparation took '+str(time.time()-start_time)+' s')


#%% Do n iterations to let the series converge
for i in range(1):


    #%% Evaluate/compute result from one step
    start_time = time.time()
    MyHelm.evalStep()
    print('Calculation took '+str(time.time()-start_time)+' s')
    
    # Display result 
    EE = np.fft.fftshift(MyHelm.psi_result)
    plt.subplot(1,2,1), plt.title('Magnitude of PSI')
    plt.imshow(np.abs(np.squeeze(EE[:,:,disp_y]))), plt.colorbar()
    plt.subplot(1,2,2), plt.title('Phase of PSI')
    plt.imshow(np.angle(np.squeeze(EE[:,:,disp_y]))), plt.colorbar()
    plt.show()
    


