# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.client import timeline

import numpy as np
import matplotlib.pyplot as plt
import h5py 
import scipy.io
import scipy as scipy

# Some helpful MATLAB functions
def abssqr(inputar):
    return np.real(inputar*np.conj(inputar))
    #return tf.abs(inp#utar)**2

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
    
def tf_abssqr(inputar):
    return tf.real(inputar*tf.conj(inputar))
    #return tf.abs(inputar)**2

def rr(inputsize_x=100, inputsize_y=100, inputsize_z=100, x_center=0, y_center = 0, z_center=0):
    x = np.linspace(-inputsize_x/2,inputsize_x/2, inputsize_x)
    y = np.linspace(-inputsize_y/2,inputsize_y/2, inputsize_y)

    if inputsize_z<=1:
        xx, yy = np.meshgrid(x+x_center, y+y_center)
        r = np.sqrt(xx**2+yy**2)
        r = np.transpose(r, [1, 0]) #???? why that?!
    else:
        z = np.linspace(-inputsize_z/2,inputsize_z/2, inputsize_z)
        xx, yy, zz = np.meshgrid(x+x_center, y+y_center, z+z_center)
        xx, yy, zz = np.meshgrid(x, y, z)
        r = np.sqrt(xx**2+yy**2+zz**2)
        r = np.transpose(r, [1, 0, 2]) #???? why that?!
        
    return np.squeeze(r)

def rr_freq(inputsize_x=100, inputsize_y=100, inputsize_z=100, x_center=0, y_center = 0, z_center=0):
    x = np.linspace(-inputsize_x/2,inputsize_x/2, inputsize_x)/inputsize_x
    y = np.linspace(-inputsize_y/2,inputsize_y/2, inputsize_y)/inputsize_y
    if inputsize_z<=1:
        xx, yy = np.meshgrid(x+x_center, y+y_center)
        r = np.sqrt(xx**2+yy**2)
        r = np.transpose(r, [1, 0]) #???? why that?!
    else:
        z = np.linspace(-inputsize_z/2,inputsize_z/2, inputsize_z)/inputsize_z
        xx, yy, zz = np.meshgrid(x+x_center, y+y_center, z+z_center)
        xx, yy, zz = np.meshgrid(x, y, z)
        r = np.sqrt(xx**2+yy**2+zz**2)
        r = np.transpose(r, [1, 0, 2]) #???? why that?!

        
    return np.squeeze(r)


def xx(inputsize_x=128, inputsize_y=128, inputsize_z=1):
    x = np.arange(-inputsize_x/2,inputsize_x/2, 1)
    y = np.arange(-inputsize_y/2,inputsize_y/2, 1)
    if inputsize_z<=1:
        xx, yy = np.meshgrid(x, y)
        xx = np.transpose(xx, [1, 0]) #???? why that?!
    else:
        z = np.arange(-inputsize_z/2,inputsize_z/2, 1)
        xx, yy, zz = np.meshgrid(x, y, z)
        xx = np.transpose(xx, [1, 0, 2]) #???? why that?!
    return np.squeeze(xx)

def yy(inputsize_x=128, inputsize_y=128, inputsize_z=1):
    x = np.arange(-inputsize_x/2,inputsize_x/2, 1)
    y = np.arange(-inputsize_y/2,inputsize_y/2, 1)
    if inputsize_z<=1:
        xx, yy = np.meshgrid(x, y)
        yy = np.transpose(yy, [1, 0]) #???? why that?!
    else:
        z = np.arange(-inputsize_z/2,inputsize_z/2, 1)
        xx, yy, zz = np.meshgrid(x, y, z)
        yy = np.transpose(yy, [1, 0, 2]) #???? why that?!
    return np.squeeze(yy)

def zz(inputsize_x=128, inputsize_y=128, inputsize_z=1):
    nx = np.arange(-inputsize_x/2,inputsize_x/2, 1)
    ny = np.arange(-inputsize_y/2,inputsize_y/2, 1)
    nz = np.arange(-inputsize_z/2,inputsize_z/2, 1)
    xxr, yyr, zzr = np.meshgrid(nx, ny, nz)
    zzr = np.transpose(zzr, [1, 0, 2]) #???? why that?!
    return (zzr)


# %% FT

# I would recommend to use this
def my_ft2d(tensor):
    """
    fftshift(fft(ifftshift(a)))
    
    Applies shifts to work with arrays whose "zero" is in the middle 
    instead of the first element.
    
    Uses standard normalization of fft unlike dip_image.
    """
    return fftshift2d(tf.fft2d(ifftshift2d(tensor)))

def my_ift2d(tensor):
    """
    fftshift(ifft(ifftshift(a)))
    
    Applies shifts to work with arrays whose "zero" is in the middle 
    instead of the first element.
    
    Uses standard normalization of ifft unlike dip_image.
    """
    return fftshift2d(tf.ifft2d(ifftshift2d(tensor)))


# fftshifts
def fftshift2d(tensor):
    """
    Shifts high frequency elements into the center of the filter.
    Works on last 2 dims of tensor (on both for size-2-tensors)
    Works only for even number of elements along these dims.
    """
    #print("Only implemented for even number of elements in each axis.")
    last_dim = len(tensor.get_shape()) - 1  # from 0 to shape-1
    top, bottom = tf.split(tensor, 2, last_dim)  # split into two along last axis
    tensor = tf.concat([bottom, top], last_dim)  # concatenates along last axis
    left, right = tf.split(tensor, 2, last_dim - 1)  # split into two along second last axis
    tensor = tf.concat([right, left], last_dim - 1)  # concatenate along second last axis
    return tensor

def ifftshift2d(tensor):
    """
    Shifts high frequency elements into the center of the filter.
    Works on last 2 dims of tensor (on both for size-2-tensors)
    Works only for even number of elements along these dims.
    """
    #print("Only implemented for even number of elements in each axis.")
    last_dim = len(tensor.get_shape()) - 1
    left, right = tf.split(tensor, 2, last_dim - 1)
    tensor = tf.concat([right, left], last_dim - 1)
    top, bottom = tf.split(tensor, 2, last_dim)
    tensor = tf.concat([bottom, top], last_dim)
    return tensor

def fftshift3d(tensor):
    """
    Shifts high frequency elements into the center of the filter.
    Works on last 3 dims of tensor (on all for size-3-tensors)
    Works only for even number of elements along these dims.
    """
    #print("Only implemented for even number of elements in each axis.")
    top, bottom = tf.split(tensor, 2, -1)
    tensor = tf.concat([bottom, top], -1)
    left, right = tf.split(tensor, 2, -2)
    tensor = tf.concat([right, left], -2)
    front, back = tf.split(tensor, 2, -3)
    tensor = tf.concat([back, front], -3)
    return tensor

def ifftshift3d(tensor):
    """
    Shifts high frequency elements into the center of the filter.
    Works on last 3 dims of tensor (on all for size-3-tensors)
    Works only for even number of elements along these dims.
    """
    #print("Only implemented for even number of elements in each axis.")
    left, right = tf.split(tensor, 2, -2)
    tensor = tf.concat([right, left], -2)
    top, bottom = tf.split(tensor, 2, -1)
    tensor = tf.concat([bottom, top], -1)
    front, back = tf.split(tensor, 2, -3)
    tensor = tf.concat([back, front], -3)
    return tensor    


def my_ft3d(tensor):
    """
    fftshift(fft(ifftshift(a)))
    
    Applies shifts to work with arrays whose "zero" is in the middle 
    instead of the first element.
    
    Uses standard normalization of fft unlike dip_image.
    """
    return fftshift3d(tf.fft3d(ifftshift3d(tensor)))

def my_ift3d(tensor):
    """
    fftshift(ifft(ifftshift(a)))
    
    Applies shifts to work with arrays whose "zero" is in the middle 
    instead of the first element.
    
    Uses standard normalization of fft unlike dip_image.
    """
    return fftshift3d(tf.ifft3d(ifftshift3d(tensor)))

class HelmholtzSolver:
    ''' psi = HelmholtzSolver(myN, mySrc, myeps, k0, startPsi, showupdate) : solves the inhomogeneous Helmholtz equation
    % myN : refractive index distribution
    % mySrc : source distribution
    % myeps : can be empty [], default: max(abs(kr2Mk02)) + 0.001;
    % k0 : inverse wavelength k0 = (2*pi)/lambda. default: smallest Nyquist sampling: k0=0.25/max(real(myN))
    % startPsi : can be the starting field (e.g. from a previous result)
    % showupdate : every how many itereations should the result be shown.
    %
    % based on 
    % http://arxiv.org/abs/1601.05997
    % communicated by Benjamin Judkewitz
    % derived from Rainer Heintzmanns Code
    '''
    
    # initialize the Helmholtz operator
    showupdate = 10
    
    def __init__(self, myN, mySrc, myeps=None, k0=None, startPsi=None, showupdate=10):
    
        # initialize Class instance
        self.showupdate = showupdate
        self.myN = myN
        self.mySrc = mySrc
        self.myeps = myeps
        self.startPsi = startPsi 
        
        # INternal step counter 
        self.step_counter = 0
        self.nsteps = 1
        
        if k0 == None:
            k0=0.25/np.max(np.real(myN));
        else:
            self.k0=k0
        
    def computeModel(self):
        # Open a new session object 
        # self.sess = tf.Session()   
        config = tf.ConfigProto()
        jit_level = 0
        if True:
            # Turns on XLA JIT compilation.
            jit_level = tf.OptimizerOptions.ON_1
        else:
            # Turns off XLA JIT compilation.
            jit_level = tf.OptimizerOptions.OFF_1
            
        config.graph_options.optimizer_options.global_jit_level = jit_level
        self.run_metadata = tf.RunMetadata()
  
  
        self.sess = tf.InteractiveSession(config=config)
        
        # Start the code
        k02 = self.k0**2;
        kr2 = self.myN**2*k02;  # n scales with k0
        kr2Mk02 = (kr2-k02)*1j
        MinEps = np.max(np.abs(kr2Mk02)) + 0.001 #; % + .001
        
        if self.myeps == None:
            self.myeps = MinEps #; % max(abs(kr2Mk02)) + 0.001; % + .001 #% myeps = .05;
        
        # 1/(p^2-k0^2-ieps)
        if(len(self.myN.shape)>2):
            rr_temp = rr_freq(self.myN.shape[0], self.myN.shape[1], self.myN.shape[2])
        else:
            rr_temp = rr_freq(self.myN.shape[0], self.myN.shape[1], 0)
    
        GreensFktFourier = 1.0 / (abssqr(rr_temp) - (k02+0j)- 1j*self.myeps);
        
        V = kr2 - k02 - 1j*self.myeps #; % V(r)
        gamma = V * 1j / self.myeps
           
            
        ## Port everything to Tensorflow 
        self.tf_mySrc = tf.constant(np.squeeze(self.mySrc))
        self.tf_GreensFktFourier = tf.constant(np.squeeze(GreensFktFourier))
        self.tf_V = tf.constant(np.squeeze(V))
        self.tf_gamma = tf.constant(np.squeeze(gamma))
    
        if(self.myN.shape[2] !=(1)):
            # case for 3D 
            if self.startPsi == None:
                self.tf_psi = self.tf_gamma * my_ift3d(my_ft3d(self.tf_mySrc * self.tf_V) * self.tf_GreensFktFourier);
            else:
                self.tf_psi = self.startPsi
            
            self.tf_convS = my_ift3d(my_ft3d(self.tf_mySrc * self.tf_V) * self.tf_GreensFktFourier); # needs to be computed only once
        else:
            if self.startPsi == None:
                self.tf_psi = self.tf_gamma * my_ift2d(my_ft2d(self.tf_mySrc * self.tf_V) * self.tf_GreensFktFourier);
            else:
                self.tf_psi = self.startPsi
            
            self.tf_convS = my_ift2d(my_ft2d(self.tf_mySrc * self.tf_V) * self.tf_GreensFktFourier); # needs to be computed only once
            
            
            
    def step(self, nsteps = 1):
        # perform one step in the iteration of convergent born series
        
        # add the steps to the internal counter
        self.nsteps = nsteps 
        print('Creating Model for '+str(self.nsteps)+' steps.')

        for i in range(self.nsteps):
            if(self.myN.shape[2] != (1)):
                # case for 3D 
                self.tf_convPsi = my_ift3d(my_ft3d(self.tf_psi * self.tf_V) * self.tf_GreensFktFourier);
            else:
                self.tf_convPsi = my_ift2d(my_ft2d(self.tf_psi * self.tf_V) * self.tf_GreensFktFourier);
            
            self.tf_UPsi = self.tf_gamma * (self.tf_convPsi - self.tf_psi + self.tf_convS)
            self.tf_UPsi = tf.stop_gradient(self.tf_UPsi)
        

            if (1): # standard way of updating
                self.tf_psi = self.tf_psi + self.tf_UPsi
                self.tf_psi = tf.stop_gradient(self.tf_psi)
            else:  #% update,but also play with Epsilon
                print("Not implemented yet")
        #        %AbsUpdate=abs(UPsi);
        #        %AbsUpdate(AbsUpdate<10)=10;
        #        psi = psi + UPsi;
        #        % Try updating epsilon
        #        myeps = MinEps + (myeps - MinEps) /2;
        #        GreensFktFourier = 1.0 ./ (abssqr(rr(size(myN),'freq')) - k02 - i*myeps);
        #        V = kr2 - k02 - i*myeps; % V(r)
        #        gamma = V .* i./myeps;       
        #        convS = ift(ft(mySrc .* V) .* GreensFktFourier);  % needs to be computed only once
        
    
    def compileGraph(self):
        print("Init operands ")
        init_op = tf.global_variables_initializer()
        print("run init")
        self.sess.run(init_op)
        
        
    def evalStep(self):
        print('Start Computing the result')
        self.step_counter += self.nsteps
        self.psi_result = self.sess.run(self.tf_psi, 
                options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
               run_metadata=self.run_metadata)
        trace = timeline.Timeline(step_stats=self.run_metadata.step_stats)
        with open('./timeline.ctf.json', 'w') as trace_file:
            trace_file.write(trace.generate_chrome_trace_format())
        
        print('Now performed '+str(self.step_counter)+' steps.')


def insertSphere(obj_shape = [100, 100, 100], obj_dim = 0.1, obj_type = 0, diameter = 1, dn = 0.1):
    ''' Function to generate a 3D RI distribution to give an artificial sample for testing the FWD system
    INPUTS:
        obj_shape - Nx, Ny, Nz e.g. [100, 100, 100]
        obj_dim - sampling in dx/dy/dz 0.1 (isotropic for now)
        obj_type - 0; One Sphere 
                 - 1; Two Spheres 
                 - 2; Two Spheres 
                 - 3; Two Spheres inside volume
                 - 4; Shepp Logan Phantom (precomputed in MATLAB 120x120x120 dim)
        diameter - 1 (parameter for diameter of sphere)
        dn - difference of RI e.g. 0.1
        
    OUTPUTS: 
        f - 3D RI distribution
            
    '''
    # one spherical object inside a volume
    f = (dn-1)*(rr(obj_shape[0], obj_shape[1], obj_shape[2])*obj_dim < diameter)+1
    return f
    
  
def insertPerfectAbsorber(myN,SlabX,SlabW=1,direction=None,k0=None,N=4):
    '''
    % myN=insertPerfectAbsorber(myN,SlabX,SlabW) : inserts a slab of refractive index material into a dataset
    % myN : dataset to insert into
    % SlabX : middle coordinate
    % SlabW : half the width of slab
    % direction : direction of absorber: 1= left to right, -1=right to left, 2:top to bottom, -2: bottom to top
    '''

    if k0==None:
        k0=0.25/np.max(np.real(myN));

    k02 = k0**2;
    
    if myN.ndim < 3:
        myN = np.expand_dims(myN, 2)
        
        
    myXX=xx(myN.shape[0], myN.shape[1], myN.shape[2])+myN.shape[0]/2
    
    if np.abs(direction) <= 1:
        myXX=xx(myN.shape[0], myN.shape[1], myN.shape[2])+myN.shape[0]/2
    else:
        myXX=yy(myN.shape[0], myN.shape[1], myN.shape[2])+myN.shape[1]/2


    alpha=0.035*100/SlabW #; % 100 slices
    if direction > 0:
        #% myN(abs(myXX-SlabX)<=SlabW)=1.0+1i*alpha*exp(xx(mysize,'corner')/mysize(1));  % increasing absorbtion
        myX=myXX-SlabX
    else:
        # %myN(abs(myXX-SlabX)<=SlabW)=1.0+1i*alpha*exp((mysize(1)-1-xx(mysize,'corner'))/mysize(1));  % increasing absorbtion
        myX=SlabX+SlabW-myXX-1
        
        
    myMask= (myX>=0) * (myX<SlabW)

    alphaX=alpha*myX[myMask]
    
    PN=0;
    for n in range(0, N+1):
        PN = PN + np.power(alphaX,n)/factorial(n)

    k2mk02 = np.power(alphaX,N-1)*abssqr(alpha)*(N-alphaX + 2*1j*k0*myX[myMask])/(PN*factorial(N))
    
    if(myN.shape[2]!=2):
        # For 2D processing
        myN[myMask] = np.expand_dims(np.sqrt((k2mk02+k02)/k02), axis=1)
    else:
        # For 3D processing        
        myN[myMask] = np.sqrt((k2mk02+k02)/k02)
            
    #np.array(myN)[0]
    
    return myN,k0


def insertSrc(mysize,myWidth=20,myOff=None, kx=0, ky=0):
    '''
    % myN=insertSrc(myN,myWidth,myOff,kx) : inserts a Gaussian source bar for the Helmholtz simulation
    % myN : refractive indext to insert into
    % myWidth : Gaussian width
    % myOff : 2-component vector for position
    % kx : determins angle of emission (default 0)
    '''
    
    
    if myOff == None:
        myOff=((101, np.floor((myN.shape[1],2)/2)))

    mySrc = np.zeros(mysize)+1j*np.zeros(mysize)
    myOffX=myOff[0]
    myOffY=myOff[1]

    if np.size(mysize) > 2:
        myOffZ=myOff[2]
        print("WARNING: Not yet implemented")
        mySrc[myOffX,:,:] = np.exp(1j*kx * (myOffY+yy(1,mysize[1],mysize[2]))) * np.exp(-abssqr(((myOffY+yy(1,mysize[1],mysize[2]))))/(2*myWidth**2))
        mySrc[myOffX,:,:] = mySrc[myOffX,:,:] * np.exp(1j*ky * (myOffZ+zz(1,mysize[1],mysize[2]))) * np.exp(-abssqr(((myOffZ+zz(1,mysize[1],mysize[2]))))/(2*myWidth**2))
    else:

        mySrc[myOffX,:] = np.exp(1j*kx * (myOffY+yy(1,mysize[1]))) * np.exp(- abssqr(myOffY+yy(1,mysize[1]))/(2*myWidth**2))
        
    return mySrc
