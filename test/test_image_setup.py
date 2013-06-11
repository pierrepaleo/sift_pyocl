#!/usr/bin/env python
import numpy, scipy.ndimage, scipy.misc
from test_image_functions import *
from test_algebra import *
'''
Unit tests become more and more difficult as we progress in the global SIFT algorithm
For a better code visibility, the setups required by kernels will be put here
'''

def local_maxmin_setup():
    border_dist = numpy.int32(5) #SIFT
    peakthresh = numpy.float32(255.0 * 0.04 / 3.0) #SIFT uses 255.0 * 0.04 / 3.0
    EdgeThresh = numpy.float32(0.06) #SIFT
    EdgeThresh0 = numpy.float32(0.08) #SIFT
    octsize = numpy.int32(1) #initially 1, then twiced at each new octave. If we want to test for octsize 2,4..., we need a resample function
    nb_keypoints = 1000 #constant size !
		
    l2 = scipy.misc.lena().astype(numpy.float32)#[100:250,100:250] #use a part of the image to fasten tests
    l = normalize_image(l2) #do not forget to normalize the image if you want to compare with sift.cpp
    width = numpy.int32(l.shape[1])
    height = numpy.int32(l.shape[0])
    
    #Blurs and DoGs pre-allocating      
    g = (numpy.zeros(6*height*width).astype(numpy.float32)).reshape(6,height,width) #vector of 6 blurs
    DOGS = numpy.zeros((5,height,width),dtype=numpy.float32) #vector of 5 DoGs
    g[0,:,:] = numpy.copy(l)
    '''
    sift.cpp pre-process
    '''
    doubleimsize = 0 #par.DoubleImSize = 0 by default
    initsigma = 1.6
    if (doubleimsize): cursigma = 1.0
    else: cursigma = 0.5
    #Convolving initial image to achieve std = initsigma = 1.6
    if (initsigma > cursigma):
        sigma = numpy.sqrt(initsigma**2-cursigma**2)
        g[0,:,:]= numpy.copy(scipy.ndimage.filters.gaussian_filter(l, sigma, mode="reflect"))

    '''
    Blurs and DoGs
    '''
    sigmaratio = 2**(1/3.0) #sift.cpp
    #sift.cpp : for a given "i", we have : increase = initsigma*(sigmaratio)^(i-1)*sqrt(sigmaratio**2 -1)
    for i in range(1,5):
        sigma = initsigma*(sigmaratio)**(i-1.0)*numpy.sqrt(sigmaratio**2 -1.0) #sift.cpp "increase" 
        g[i] = numpy.copy(scipy.ndimage.filters.gaussian_filter(g[i-1], sigma, mode="reflect")) #blur[i]
   
    for s in range(1,5): DOGS[s-1] = -(g[s]-g[s-1]) #DoG[s-1]
     
    return border_dist, peakthresh, EdgeThresh, EdgeThresh0, octsize, nb_keypoints, width, height, DOGS, g




def interpolation_setup():
    '''
    Provides the values required by "test_interpolation"
    Previous step: local extrema detection - we got a vector of keypoints to be interpolated
    '''
    
    border_dist, peakthresh, EdgeThresh, EdgeThresh0, octsize, nb_keypoints, width, height, DOGS, g = local_maxmin_setup()
    
    s = numpy.int32(1) #1, 2 or 3... NOT 0 NOR 4 !
    nb_keypoints = numpy.int32(nb_keypoints)

    #Assumes that local_maxmin is working so that we can use Python's "my_local_maxmin" instead of the kernel
    keypoints_prev, actual_nb_keypoints = my_local_maxmin(DOGS, peakthresh,border_dist, octsize, 
        EdgeThresh0, EdgeThresh,nb_keypoints,s,width,height)

    return border_dist, peakthresh, EdgeThresh, EdgeThresh0, octsize, nb_keypoints, actual_nb_keypoints, width, height, DOGS, s, keypoints_prev, g[s]
    
    
    
    
def orientation_setup():
    '''
    Provides the values required by "test_orientation"
    Previous step: interpolation - we got a vector of valid keypoints
    '''
    border_dist, peakthresh, EdgeThresh, EdgeThresh0, octsize, nb_keypoints, actual_nb_keypoints, width, height, DOGS, s, keypoints_prev, blur = interpolation_setup()
    
    #actual_nb_keypoints = numpy.int32(len((keypoints_prev[:,0])[keypoints_prev[:,1] != -1]))
    ref = numpy.copy(keypoints_prev)
    #There are actually less than "actual_nb_keypoints" keypoints ("holes" in the vector), but we can use it as a boundary
    for i,k in enumerate(ref[:actual_nb_keypoints,:]):
        ref[i]= my_interp_keypoint(DOGS, s, k[1], k[2],5,peakthresh,width,height)
    
    grad, ori = my_gradient(blur) #gradient is applied on blur[s]
   # ref, actual_nb_keypoints = my_compact(ref,nb_keypoints)
    

    return ref, nb_keypoints, actual_nb_keypoints, grad, ori, octsize
    
    
