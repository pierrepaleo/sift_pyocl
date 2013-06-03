#!/usr/bin/env python
# -*- coding: utf8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/kif/sift_pyocl
#

"""
Test suite for algebra kernels
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
import scipy, scipy.misc, scipy.ndimage #, pylab
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

print "working on %s" % ctx.devices[0].name

def my_gradient(mat):
    """
    numpy implementation of gradient :
    "The gradient is computed using central differences in the interior and first differences at the boundaries. The returned gradient hence has the same shape as the input array."
    """
    g = numpy.gradient(mat)
    return numpy.sqrt(g[0]**2+g[1]**2), numpy.arctan2(g[0],g[1]) #image.cl/compute_gradient_orientation() puts a "-" here
    
    
def my_local_maxmin(dog_prev,dog,dog_next,thresh,border_dist):
    """
    a python implementation of 3x3 maximum (positive values) or minimum (negative or null values) detection
    an extremum candidate "val", read in the  has to be greater than 0.8*thresh
    The three DoG have the same size.
    """
    output = numpy.zeros_like(dog)
    width = dog.shape[1]
    height = dog.shape[0]
    
    for j in range(border_dist,width - border_dist):
        for i in range(border_dist,height - border_dist):
            val = dog[i,j]
            if (numpy.abs(val) > 0.8*thresh):
                output[i,j] = is_maxmin(dog_prev,dog,dog_next,val,i,j)
    return output
    
    
def is_maxmin(dog_prev,dog,dog_next,val,i0,j0):
    """
    return 1 iff mat[i0,j0] is a local (3x3) maximum
    return -1 iff mat[i0,j0] is a local (3x3) minimum
    return 0 by default (neither maximum nor minimum)
     * Assumes that we are not on the edges, i.e border_dist >= 2 above
    """
    ismax = 0
    ismin = 0
    if (val > 0.0): ismax = 1
    else: ismin = 1
    for j in range(j0-1,j0+1+1):
        for i in range(i0-1,i0+1+1):
            if (ismax == 1):
                if (dog_prev[i,j] > val or dog[i,j] > val or dog_next[i,j] > val): ismax = 0
            if (ismin == 1):
                if (dog_prev[i,j] < val or dog[i,j] < val or dog_next[i,j] < val): ismin = 0;
            
    if (ismax == 1): return 1 
    if (ismin == 1): return -1
    return 0
    
    

class test_image(unittest.TestCase):
    def setUp(self):
    
        kernel_path = os.path.join(os.path.dirname(os.path.abspath(sift.__file__)), "image.cl")
        kernel_src = open(kernel_path).read()
        self.program = pyopencl.Program(ctx, kernel_src).build()
        self.wg = (2, 256)
        

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
        self.peakthresh = numpy.float32(255.0 * 0.04 / 3.0) #SIFT

        l = scipy.misc.lena().astype(numpy.float32)
        self.width = numpy.int32(l.shape[1])
        self.height = numpy.int32(l.shape[0])

        #the DoG should be computed with *our* kernels
        g = (numpy.zeros(4*self.height*self.width).astype(numpy.float32)).reshape(4,self.height,self.width) #vector of 4 images
        sigma=1.6 #par.InitSigma
        g[0,:,:]= numpy.copy(scipy.ndimage.filters.gaussian_filter(l, sigma, mode="reflect"))
        for i in range(1,4):
            sigma = sigma*(2.0**(1.0/5.0)) #SIFT
            g[i] = numpy.copy(scipy.ndimage.filters.gaussian_filter(l, sigma, mode="reflect"))

        self.dog_prev = g[1]-g[0]
        self.dog = g[2]-g[1]
        self.dog_next = g[3]-g[2]

        self.gpu_dog_prev = pyopencl.array.to_device(queue, self.dog_prev)
        self.gpu_dog = pyopencl.array.to_device(queue, self.dog)
        self.gpu_dog_next = pyopencl.array.to_device(queue, self.dog_next)
        self.output = pyopencl.array.empty(queue, self.dog.shape, dtype=numpy.int32, order="C")
        self.shape = calc_size(self.dog.shape, self.wg)

        t0 = time.time()
        k1 = self.program.local_maxmin(queue, self.shape, self.wg, self.gpu_dog_prev.data, self.gpu_dog.data, self.gpu_dog_next.data, self.output.data, self.width, self.height, self.border_dist, self.peakthresh)
        res = self.output.get()
        t1 = time.time()
        ref = my_local_maxmin(self.dog_prev,self.dog,self.dog_next,self.peakthresh,self.border_dist)
        t2 = time.time()
        delta = abs(ref - res).max()
        
        '''
        fig = pylab.figure()
        sp1 = fig.add_subplot(221)
        sh1 = sp1.imshow(res, interpolation="nearest")
        sp2 = fig.add_subplot(222)
        sh2 = sp2.imshow(ref,interpolation="nearest")
        cbar = fig.colorbar(sh2)
        fig.show()
        raw_input("enter")
        '''

        self.assert_(delta == 0, "delta=%s" % (delta)) #as the matrices contain integers, "delta == 0" can be used
        logger.info("delta=%s" % delta)
        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Local extrema search took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
  



def test_suite_image():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_image("test_gradient"))
    testSuite.addTest(test_image("test_local_maxmin"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_image()
    runner = unittest.TextTestRunner()
    if not runner.run(mysuite).wasSuccessful():
        sys.exit(1)

