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
import scipy, scipy.misc, scipy.ndimage
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




def my_combine(mat1,a1,mat2,a2):
    """
    reference linear combination
    """
    return a1*mat1+a2*mat2
    
    
    
def my_compact(keypoints,nbkeypoints):
    '''
    Reference compacting
    '''
    output = -numpy.ones_like(keypoints,dtype=numpy.float32)
    counter = 0
    idx = numpy.where(keypoints[:,1]!=-1)[0]
    length = idx.size
    output[:length,0] = keypoints[idx,2]
    output[:length,1] = keypoints[idx,1]
    output[:length,2] = keypoints[idx,3]
    output[:length,3] = 0
#    for i in range(0,nbkeypoints):
#        if (keypoints[i,1] != -1 and keypoints[i,0] != 0):
#            output[counter]= keypoints[i,2],keypoints[i,1],keypoints[i,3],0.0
#            counter+=1
    return output, counter




class test_algebra(unittest.TestCase):
    def setUp(self):
    	
        kernel_path = os.path.join(os.path.dirname(os.path.abspath(sift.__file__)), "algebra.cl")
        kernel_src = open(kernel_path).read()
        self.program = pyopencl.Program(ctx, kernel_src).build()
        self.wg = (1, 512)
       

    def tearDown(self):
        self.mat1 = None
        self.mat2 = None
        self.program = None
        
        
        
        
        
        
    def test_combine(self):
        """
        tests the combine (linear combination) kernel
        """
        self.width = numpy.int32(15)
    	self.height = numpy.int32(14)
    	self.coeff1 = numpy.random.rand(1)[0].astype(numpy.float32)
    	self.coeff2 = numpy.random.rand(1)[0].astype(numpy.float32)
    	self.mat1 = numpy.random.rand(self.height,self.width).astype(numpy.float32)
    	self.mat2 = numpy.random.rand(self.height,self.width).astype(numpy.float32)
    	
        self.gpu_mat1 = pyopencl.array.to_device(queue, self.mat1)
        self.gpu_mat2 = pyopencl.array.to_device(queue, self.mat2)
        self.gpu_out = pyopencl.array.empty(queue, self.mat1.shape, dtype=numpy.float32, order="C")
        self.shape = calc_size(self.mat1.shape, self.wg)

        t0 = time.time()
        k1 = self.program.combine(queue, self.shape, self.wg, 
                                  self.gpu_mat1.data, self.coeff1, self.gpu_mat2.data, self.coeff2, 
                                  self.gpu_out.data, numpy.int32(0),
                                  self.width, self.height)
        res = self.gpu_out.get()
        t1 = time.time()
        ref = my_combine(self.mat1,self.coeff1,self.mat2,self.coeff2)
        t2 = time.time()
        delta = abs(ref - res).max()

        self.assert_(delta < 1e-4, "delta=%s" % (delta))
        logger.info("delta=%s" % delta)
        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Linear combination took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))



    def test_compact(self):
        """
        tests the "compact" kernel
        """
        
        nbkeypoints = 1000 #constant value
        keypoints = numpy.random.rand(nbkeypoints,4).astype(numpy.float32)
        for i in range(0,nbkeypoints):
            if ((numpy.random.rand(1))[0] < 0.75):
                keypoints[i]=(-1,-1,-1,-1)
        
        self.gpu_keypoints = pyopencl.array.to_device(queue, keypoints)
        self.output = pyopencl.array.empty(queue, (nbkeypoints,4), dtype=numpy.float32, order="C")
        self.output.fill(-1.0,queue)
        self.counter = pyopencl.array.zeros(queue, (1,), dtype=numpy.int32, order="C")
        wg = max(self.wg),
        shape = calc_size((keypoints.shape[0],), wg)
        nbkeypoints = numpy.int32(nbkeypoints)
        
        t0 = time.time()
        k1 = self.program.compact(queue, shape, wg, 
        	self.gpu_keypoints.data, self.output.data, self.counter.data, nbkeypoints)
        res = self.output.get()
        count = self.counter.get()[0]
        t1 = time.time()
        ref, count_ref = my_compact(keypoints,nbkeypoints)
       
        t2 = time.time()

        res_sort_arg = res[:,0].argsort(axis=0)     
        res_sort = res[res_sort_arg]
        ref_sort_arg = ref[:,0].argsort(axis=0)     
        ref_sort = ref[ref_sort_arg]
        delta = abs((res_sort - ref_sort)).max()
        self.assert_(delta < 1e-5, "delta=%s" % (delta))
        logger.info("delta=%s" % delta)
        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Compact operation took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))















  

def test_suite_algebra():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_algebra("test_combine"))
    testSuite.addTest(test_algebra("test_compact"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_algebra()
    runner = unittest.TextTestRunner()
    if not runner.run(mysuite).wasSuccessful():
        sys.exit(1)

