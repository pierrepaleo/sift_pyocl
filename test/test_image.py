#!/usr/bin/env python
# -*- coding: utf8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/kif/sift_pyocl
#

"""
Test suite for image kernels
"""

from __future__ import division

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "BSD"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "2013-05-28"
__license__ = """
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

"""

import time, os, logging
import numpy
import pyopencl, pyopencl.array
import scipy, scipy.misc, scipy.ndimage, pylab
import sys
import unittest
from utilstest import UtilsTest, getLogger, ctx
import sift
from sift.utils import calc_size
logger = getLogger(__file__)
if logger.getEffectiveLevel() <= logging.INFO:
    PROFILE = True
    queue = pyopencl.CommandQueue(ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
    import pylab
else:
    PROFILE = False
    queue = pyopencl.CommandQueue(ctx)
    
SHOW_FIGURES = False
    

print "working on %s" % ctx.devices[0].name

def my_gradient(mat):
    """
    numpy implementation of gradient :
    "The gradient is computed using central differences in the interior and first differences at the boundaries. The returned gradient hence has the same shape as the input array."
    """
    g = numpy.gradient(mat)
    return numpy.sqrt(g[0]**2+g[1]**2), numpy.arctan2(g[0],g[1]) #image.cl/compute_gradient_orientation() puts a "-" here
    
    
def my_local_maxmin(dog_prev,dog,dog_next,thresh,border_dist,octsize,EdgeThresh0,EdgeThresh,nb_keypoints,s):
    """
    a python implementation of 3x3 maximum (positive values) or minimum (negative or null values) detection
    an extremum candidate "val" has to be greater than 0.8*thresh
    The three DoG have the same size.
    """
    output = numpy.zeros((nb_keypoints,4),dtype=numpy.float32)
    width = dog.shape[1]
    height = dog.shape[0]
    counter = 0
    
    for j in range(border_dist,width - border_dist):
        for i in range(border_dist,height - border_dist):
            val = dog[i,j]
            if (numpy.abs(val) > 0.8*thresh): #keypoints refinement: eliminating low-contrast points
                if (is_maxmin(dog_prev,dog,dog_next,val,i,j,octsize,EdgeThresh0,EdgeThresh) != 0):
                	output[counter,0]=val
                	output[counter,1]=i
                	output[counter,2]=j
                	output[counter,3]=s
                	counter+=1   	      	
    return output
    
    
def is_maxmin(dog_prev,dog,dog_next,val,i0,j0,octsize,EdgeThresh0,EdgeThresh):
    """
    return 1 iff mat[i0,j0] is a local (3x3) maximum
    return -1 iff mat[i0,j0] is a local (3x3) minimum
    return 0 by default (neither maximum nor minimum, or value on an edge)
     * Assumes that we are not on the edges, i.e border_dist >= 2 above
    """
    ismax = 0
    ismin = 0
    res = 0
    if (val > 0.0): ismax = 1
    else: ismin = 1
    for j in range(j0-1,j0+1+1):
        for i in range(i0-1,i0+1+1):
            if (ismax == 1):
                if (dog_prev[i,j] > val or dog[i,j] > val or dog_next[i,j] > val): ismax = 0
            if (ismin == 1):
                if (dog_prev[i,j] < val or dog[i,j] < val or dog_next[i,j] < val): ismin = 0;
    
    if (ismax == 1): res =  1 
    if (ismin == 1): res = -1
    
    #keypoint refinement: eliminating points at edges
    H00 = dog[i0-1,j0] - 2.0 * dog[i0,j0] + dog[i0+1,j0]
    H11 = dog[i0,j0-1]- 2.0 * dog[i0,j0] + dog[i0,j0+1]
    H01 = ( (dog[i0+1,j0+1] - dog[i0+1,j0-1])
		- (dog[i0-1,j0+1] - dog[i0-1,j0-1]) ) / 4.0;

    det = H00 * H11 - H01 * H01
    trace = H00 + H11

    if (octsize <= 1):
        thr = EdgeThresh0
    else:
        thr = EdgeThresh
    if (det < thr * trace * trace):
        res = 0
        
    return res
    
    

def my_create_keypoints(peaks,nb_keypoints,s):
    '''
    create a vector of nb_keypoints from the peaks matrix
    '''
    height, width = peaks.shape
    output = numpy.zeros((nb_keypoints,4),dtype=numpy.float32)
    
    b = (peaks != 0)
    nb = b.sum()
    tmp = numpy.where(b)
    output[:nb,1],output[:nb,2] = tmp
    output[:nb,0] = peaks[tmp]
    output[:nb,3] = s
    
    return output
    
    
    
    

    
    

class test_image(unittest.TestCase):
    def setUp(self):
    
        kernel_path = os.path.join(os.path.dirname(os.path.abspath(sift.__file__)), "image.cl")
        kernel_src = open(kernel_path).read()
        self.program = pyopencl.Program(ctx, kernel_src).build()
        self.wg = (1, 512)
        

    def tearDown(self):
        self.mat = None
        self.program = None
        
        
        
        
        
        
    def test_gradient(self):
        """
        tests the gradient kernel (norm and orientation)
        """

        self.width = numpy.int32(15)
        self.height = numpy.int32(14)

        self.mat = numpy.random.rand(self.height,self.width).astype(numpy.float32)
        self.gpu_mat = pyopencl.array.to_device(queue, self.mat)
        self.gpu_grad = pyopencl.array.empty(queue, self.mat.shape, dtype=numpy.float32, order="C")
        self.gpu_ori = pyopencl.array.empty(queue, self.mat.shape, dtype=numpy.float32, order="C")
        self.shape = calc_size(self.mat.shape, self.wg)

        t0 = time.time()
        k1 = self.program.compute_gradient_orientation(queue, self.shape, self.wg, self.gpu_mat.data, self.gpu_grad.data, self.gpu_ori.data, self.width, self.height)
        res_norm = self.gpu_grad.get()
        res_ori = self.gpu_ori.get()
        t1 = time.time()
        ref_norm,ref_ori = my_gradient(self.mat)
        t2 = time.time()
        delta_norm = abs(ref_norm - res_norm).max()
        delta_ori = abs(ref_ori - res_ori).max()
        self.assert_(delta_norm < 1e-4, "delta_norm=%s" % (delta_norm))
        self.assert_(delta_ori < 1e-4, "delta_ori=%s" % (delta_ori))
        logger.info("delta_norm=%s" % delta_norm)
        logger.info("delta_ori=%s" % delta_ori)
        
        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Gradient computation took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))





    def test_local_maxmin(self):
        """
        tests the local maximum/minimum detection kernel
        """
        self.border_dist = numpy.int32(5) #SIFT
        self.peakthresh = numpy.float32(0.01)#(255.0 * 0.04 / 3.0) #SIFT
        self.EdgeThresh = numpy.float32(0.06) #SIFT
        self.EdgeThresh0 = numpy.float32(0.08) #SIFT
        self.octsize = numpy.int32(4) #initially 1, then twiced at each new octave
        self.nb_keypoints = 10000 #constant size !
		
		
        l = scipy.misc.lena().astype(numpy.float32)[100:250,100:250]
        self.width = numpy.int32(l.shape[1])
        self.height = numpy.int32(l.shape[0])

        g = (numpy.zeros(4*self.height*self.width).astype(numpy.float32)).reshape(4,self.height,self.width) #vector of 4 images
        sigma=1.6 #par.InitSigma
        g[0,:,:]= numpy.copy(scipy.ndimage.filters.gaussian_filter(l, sigma, mode="reflect"))
        for i in range(1,4):
            sigma = sigma*(2.0**(1.0/5.0)) #SIFT
            g[i] = numpy.copy(scipy.ndimage.filters.gaussian_filter(l, sigma, mode="reflect"))

        self.dog_prev = g[1]-g[0]
        self.dog = g[2]-g[1]
        self.dog_next = g[3]-g[2]
        self.s = numpy.float32(1.0) #0, 1, 2 or 3... 1 here
        
        self.gpu_dog_prev = pyopencl.array.to_device(queue, self.dog_prev)
        self.gpu_dog = pyopencl.array.to_device(queue, self.dog)
        self.gpu_dog_next = pyopencl.array.to_device(queue, self.dog_next)
        #self.output = pyopencl.array.empty(queue, self.dog.shape, dtype=numpy.float32, order="C")
        self.output = pyopencl.array.zeros(queue, (self.nb_keypoints,4), dtype=numpy.float32, order="C")
        self.counter = pyopencl.array.zeros(queue, (1,), dtype=numpy.int32, order="C")
        self.nb_keypoints = numpy.int32(self.nb_keypoints)
        self.shape = calc_size(self.dog.shape, self.wg)
        
        t0 = time.time()
        k1 = self.program.local_maxmin(queue, self.shape, self.wg, 
        	self.gpu_dog_prev.data, self.gpu_dog.data, self.gpu_dog_next.data, self.output.data, 
       		self.border_dist, self.peakthresh, self.octsize, self.EdgeThresh0, self.EdgeThresh,
       		self.counter.data, self.nb_keypoints, self.s,
       		self.width, self.height)
        
        res = self.output.get()
        t1 = time.time()
        ref = my_local_maxmin(self.dog_prev,self.dog,self.dog_next,
        	self.peakthresh,self.border_dist, self.octsize, self.EdgeThresh0, self.EdgeThresh,self.nb_keypoints,self.s)
        t2 = time.time()
        
        #we have to sort the arrays, for peaks orders is unknown for GPU
        res_peaks = res[(res[:,0].argsort(axis=0)),0]
        ref_peaks = ref[(ref[:,0].argsort(axis=0)),0]
        res_r = res[(res[:,1].argsort(axis=0)),1]
        ref_r = ref[(ref[:,1].argsort(axis=0)),1]
        res_c = res[(res[:,2].argsort(axis=0)),2]
        ref_c = ref[(ref[:,2].argsort(axis=0)),2]
        #res_s = res[(res[:,3].argsort(axis=0)),3]
        #ref_s = ref[(ref[:,3].argsort(axis=0)),3]
        
        delta_peaks = abs(ref_peaks - res_peaks).max()
        delta_r = abs(ref_r - res_r).max()
        delta_c = abs(ref_c - res_c).max()
               
        self.assert_(delta_peaks < 1e-4, "delta_peaks=%s" % (delta_peaks))
        self.assert_(delta_r < 1e-4, "delta_r=%s" % (delta_r))
        self.assert_(delta_c < 1e-4, "delta_c=%s" % (delta_c))
        logger.info("delta_peaks=%s" % delta_peaks)
        logger.info("delta_r=%s" % delta_r)
        logger.info("delta_c=%s" % delta_c)

        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Local extrema search took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
  




    def test_create_keypoints(self):
        """
        tests the create_keypoints kernel
        OLD function, useless now
        """
        self.width = numpy.int32(150)
        self.height = numpy.int32(146)
        self.nb_keypoints = 10000 #constant size !
        self.s = numpy.float32(1.0) #0, 1, 2 or 3
        #fill with zeros
        self.peaks = numpy.random.rand(self.height,self.width).astype(numpy.float32)
        thres = (self.peaks <= 0.7)
        self.peaks[thres] = 0
        
        self.gpu_peaks = pyopencl.array.to_device(queue, self.peaks)
        self.output = pyopencl.array.zeros(queue, (self.nb_keypoints,4), dtype=numpy.float32, order="C")
        self.counter = pyopencl.array.zeros(queue, (1,), dtype=numpy.int32, order="C")
        self.shape = calc_size(self.peaks.shape, self.wg)
        self.nb_keypoints = numpy.int32(self.nb_keypoints)
        
        t0 = time.time()
        k1 = self.program.create_keypoints(queue, self.shape, self.wg,
        	self.gpu_peaks.data, self.s, self.output.data, self.nb_keypoints, self.counter.data, self.width, self.height)
        res = self.output.get()
        t1 = time.time()
        ref = my_create_keypoints(self.peaks,self.nb_keypoints,self.s)
        t2 = time.time()
        
        #we have to sort the arrays, for peaks orders is unknown for GPU
        res_peaks = res[(res[:,0].argsort(axis=0)),0]
        ref_peaks = ref[(ref[:,0].argsort(axis=0)),0]
        res_r = res[(res[:,1].argsort(axis=0)),1]
        ref_r = ref[(ref[:,1].argsort(axis=0)),1]
        res_c = res[(res[:,2].argsort(axis=0)),2]
        ref_c = ref[(ref[:,2].argsort(axis=0)),2]
        #res_s = res[(res[:,3].argsort(axis=0)),3]
        #ref_s = ref[(ref[:,3].argsort(axis=0)),3]
        
        delta_peaks = abs(ref_peaks - res_peaks).max()
        delta_r = abs(ref_r - res_r).max()
        delta_c = abs(ref_c - res_c).max()
               
        self.assert_(delta_peaks < 1e-4, "delta_peaks=%s" % (delta_peaks))
        self.assert_(delta_r < 1e-4, "delta_r=%s" % (delta_r))
        self.assert_(delta_c < 1e-4, "delta_c=%s" % (delta_c))
        logger.info("delta_peaks=%s" % delta_peaks)
        logger.info("delta_r=%s" % delta_r)
        logger.info("delta_c=%s" % delta_c)
        
        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Keypoints vector creation took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))









def test_suite_image():
    testSuite = unittest.TestSuite()
    #testSuite.addTest(test_image("test_gradient"))
    testSuite.addTest(test_image("test_local_maxmin"))
    #testSuite.addTest(test_image("test_create_keypoints"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_image()
    runner = unittest.TextTestRunner()
    if not runner.run(mysuite).wasSuccessful():
        sys.exit(1)

