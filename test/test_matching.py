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
from test_image_functions import * #for Python implementation of tested functions
from test_image_setup import *
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
PRINT_KEYPOINTS = False
USE_CPU = False
USE_CPP_SIFT = False #use reference cplusplus implementation for descriptors comparison... not valid for (octsize,scale)!=(1,1)



print "working on %s" % ctx.devices[0].name

'''
For Python implementation of tested functions, see "test_image_functions.py"
'''



class test_matching(unittest.TestCase):
    def setUp(self):

        kernel_path = os.path.join(os.path.dirname(os.path.abspath(sift.__file__)), "matching.cl")
        kernel_src = open(kernel_path).read()
        self.program = pyopencl.Program(ctx, kernel_src).build()
        self.wg = (1, 128)



    def tearDown(self):
        self.mat = None
        self.program = None





    def test_matching(self):
        '''
        tests keypoints matching kernel
        '''    
        kp1, kp2, nb_keypoints, actual_nb_keypoints = matching_setup()
        keypoints_start, keypoints_end = 0, actual_nb_keypoints
        ratio_th = numpy.float32(0.5329) #sift.cpp : 0.73*0.73
        print("Working on keypoints : [%s,%s]" % (keypoints_start, keypoints_end-1))
        
#        if (USE_CPU):
#            print "Using CPU-optimized kernels"
        wg = 1,
        shape = kp1.shape[0]*wg[0], #TODO: bound for the min size of kp1, kp2
#        else:
#            wg = (4, 4, 8)
#            shape = int(keypoints.shape[0]*wg[0]), 4, 8
        
        gpu_keypoints1 = pyopencl.array.to_device(queue, kp1)
        gpu_keypoints2 = pyopencl.array.to_device(queue, kp2)
        gpu_matchings = pyopencl.array.zeros(queue, (keypoints_end-keypoints_start,2),dtype=numpy.uint32, order="C")
        keypoints_start, keypoints_end = numpy.int32(keypoints_start), numpy.int32(keypoints_end)
        counter = pyopencl.array.zeros(queue, (1,1),dtype=numpy.int32, order="C")

        t0 = time.time()
        k1 = self.program.matching(queue, shape, wg,
        		gpu_keypoints1.data, gpu_keypoints2.data, gpu_matchings.data, counter.data,
        		nb_keypoints, ratio_th, keypoints_start, keypoints_end)
        res = gpu_matchings.get()
        cnt = counter.get()
        t1 = time.time()

        if (USE_CPP_SIFT):
            import feature
            sc = feature.SiftAlignment()
            ref2 = sc.sift(scipy.misc.lena()) #ref2.x, ref2.y, ref2.scale, ref2.angle, ref2.desc --- ref2[numpy.argsort(ref2.y)]).desc
            ref = ref2.desc
        else:
            ref, nb_match = my_matching(kp1, kp2, keypoints_start, keypoints_end)

        t2 = time.time()
        
        res_sort = res[numpy.argsort(res[:,1])]
        ref_sort = ref[numpy.argsort(ref[:,1])]
        
        print res_sort[0:20]
        print ""
        print ref_sort[0:20]
        print("OpenCL: %d match / Python: %d match" %(cnt,nb_match))
        
        if not(USE_CPP_SIFT):
            import feature
            sc = feature.SiftAlignment()
            ref_sift = sc.sift(scipy.misc.lena())
            ref_sift_2 = ref_sift[::-1]
            siftmatch = feature.sift_match(ref_sift, ref_sift_2)
        print siftmatch[0:10]


        #sort to compare added keypoints
        '''
        delta = abs(res_sort-ref_sort).max()
        self.assert_(delta == 0, "delta=%s" % (delta)) #integers
        logger.info("delta=%s" % delta)
        '''

        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Matching took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
            
            
            
            

def test_suite_matching():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_matching("test_matching"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_matching()
    runner = unittest.TextTestRunner()
    if not runner.run(mysuite).wasSuccessful():
        sys.exit(1)

