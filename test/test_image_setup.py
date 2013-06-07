#!/usr/bin/env python
import numpy, scipy.misc
from test_image_functions import *
'''
Unit tests become more and more difficult as we progress in the global SIFT algorithm
For a better code visibility, the setups required by kernels will be put here
'''




def interpolation_setup():
    '''
    Provides the values required by "test_interpolation"
    Previous step: local extrema detection - we got a vector of keypoints to be interpolated
    '''
    border_dist = numpy.int32(5) #SIFT
    peakthresh = numpy.float32(255.0 * 0.04 / 3.0) #SIFT - Take a lower threshold for more keypoints
    EdgeThresh = numpy.float32(0.06) #SIFT
    EdgeThresh0 = numpy.float32(0.08) #SIFT
    octsize = numpy.int32(4) #initially 1, then twiced at each new octave
    nb_keypoints = 10000 #constant size !
    
    l = scipy.misc.lena().astype(numpy.float32)#[100:250,100:250] #take a part of the image to fasten the tests
    width = numpy.int32(l.shape[1])
    height = numpy.int32(l.shape[0])

    g = (numpy.zeros(4*height*width).astype(numpy.float32)).reshape(4,height,width)
    sigma=1.6 #par.InitSigma
    g[0,:,:]= numpy.copy(scipy.ndimage.filters.gaussian_filter(l, sigma, mode="reflect")) #blur[0]
    for i in range(1,4):
        sigma = sigma*(2.0**(1.0/5.0)) #SIFT
        g[i] = numpy.copy(scipy.ndimage.filters.gaussian_filter(l, sigma, mode="reflect")) #blur[i]

    #DOGS pre-allocating      
    DOGS = numpy.zeros((4,height,width),dtype=numpy.float32)
    for s in range(1,4): DOGS[s-1] = g[s]-g[s-1] #DoG[s-1]
    s = numpy.int32(1) #1, 2 or 3... NOT 0 NOR 4 !

    nb_keypoints = numpy.int32(nb_keypoints)

    #Assumes that local_maxmin is working so that we can use Python's "my_local_maxmin" instead of the kernel
    keypoints_prev = my_local_maxmin(DOGS, peakthresh,border_dist, octsize, 
        EdgeThresh0, EdgeThresh,nb_keypoints,s,width,height)

    return border_dist, peakthresh, EdgeThresh, EdgeThresh0, octsize, 
    	nb_keypoints, width, height, DOGS, s, keypoints_prev, g[s]
    
    
    
    
def orientation_setup():
    '''
    Provides the values required by "test_orientation"
    Previous step: interpolation - we got a vector of valid keypoints
    '''
    
    border_dist, peakthresh, EdgeThresh, EdgeThresh0, octsize, 
        	nb_keypoints, width, height, DOGS, s, actual_nb_keypoints, blur = interpolation_setup()
        	
    ref = numpy.copy(keypoints_prev) #important here
        for i,k in enumerate(ref[:actual_nb_keypoints,:]):
            ref[i]= my_interp_keypoint(DOGS, s, k[1], k[2],5,peakthresh,width,height)
    
    grad, ori = my_gradient(blur) #gradient is applied on blur[s]

    return ref, nb_keypoints, actual_nb_keypoints, grad, ori, octsize
    
    
    
