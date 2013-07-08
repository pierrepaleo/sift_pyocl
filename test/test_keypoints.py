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


print "working on %s" % ctx.devices[0].name

'''
For Python implementation of tested functions, see "test_image_functions.py"
'''



class test_keypoints(unittest.TestCase):
    def setUp(self):

        kernel_path = os.path.join(os.path.dirname(os.path.abspath(sift.__file__)), "keypoints.cl")
        kernel_src = open(kernel_path).read()
        self.program = pyopencl.Program(ctx, kernel_src).build()
        self.wg = (1, 128)



    def tearDown(self):
        self.mat = None
        self.program = None




    def test_orientation(self):
        '''
        #tests keypoints orientation assignment kernel
        '''

        #orientation_setup :
        keypoints, nb_keypoints, updated_nb_keypoints, grad, ori, octsize = orientation_setup()
        #keypoints is a compacted vector of keypoints #not anymore
        keypoints_before_orientation = numpy.copy(keypoints) #important here
        wg = 128, #FIXME : have to choose it for histograms #wg = max(self.wg),
        shape = keypoints.shape[0]*wg[0],
        #shape = calc_size(keypoints.shape, self.wg)
        gpu_keypoints = pyopencl.array.to_device(queue, keypoints)
        actual_nb_keypoints = numpy.int32(updated_nb_keypoints)
        print("Max. number of keypoints before orientation assignment : %s" % actual_nb_keypoints)

        gpu_grad = pyopencl.array.to_device(queue, grad)
        gpu_ori = pyopencl.array.to_device(queue, ori)
        orisigma = numpy.float32(1.5) #SIFT
        grad_height, grad_width = numpy.int32(grad.shape)
        keypoints_start = numpy.int32(0)
        keypoints_end = numpy.int32(actual_nb_keypoints)
        counter = pyopencl.array.to_device(queue, keypoints_end) #actual_nb_keypoints)

        t0 = time.time()
        k1 = self.program.orientation_assignment(queue, shape, wg,
        	gpu_keypoints.data, gpu_grad.data, gpu_ori.data, counter.data,
        	octsize, orisigma, nb_keypoints, keypoints_start, keypoints_end, grad_width, grad_height)
        res = gpu_keypoints.get()
        cnt = counter.get()
        t1 = time.time()

        ref, updated_nb_keypoints = my_orientation(keypoints, nb_keypoints, keypoints_start, keypoints_end, grad, ori, octsize, orisigma)

        t2 = time.time()

        if (PRINT_KEYPOINTS):
            print("Keypoints after orientation assignment :")
            print res[0:actual_nb_keypoints]#[0:10]
            print " "
            print ref[0:7]

        print("Total keypoints for kernel : %s -- For Python : %s \t [octsize = %s]" % (cnt, updated_nb_keypoints, octsize))

        
        #sort to compare added keypoints
        d1, d2, d3, d4 = keypoints_compare(ref[0:cnt], res[0:updated_nb_keypoints]) #FIXME: our sift finds one additional keypoint for "lena".
        self.assert_(d1 < 1e-4, "delta_cols=%s" % (d1))
        self.assert_(d2 < 1e-4, "delta_rows=%s" % (d2))
        self.assert_(d3 < 1e-4, "delta_sigma=%s" % (d3))
        self.assert_(d4 < 1e-4, "delta_angle=%s" % (d4))
        logger.info("delta_cols=%s" % d1)
        logger.info("delta_rows=%s" % d2)
        logger.info("delta_sigma=%s" % d3)
        logger.info("delta_angle=%s" % d4)
        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Orientation assignment took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
        

    def test_descriptor(self):
        '''
        #tests keypoints descriptors creation kernel
        '''

        #descriptor_setup :
        keypoints_o, nb_keypoints, actual_nb_keypoints, grad, ori, octsize = descriptor_setup()
        #keypoints should be a compacted vector of keypoints
        keypoints_start, keypoints_end = 0, actual_nb_keypoints
        #keypoints_start, keypoints_end = 20, 30
        keypoints = keypoints_o[keypoints_start:keypoints_end]
        print("Working on keypoints : [%s,%s] (octave = %s)" % (keypoints_start, keypoints_end-1,int(numpy.log2(octsize)+1)))
        wg = 4,4,8 #FIXME : have to choose it for histograms #wg = max(self.wg),
        shape = keypoints.shape[0]*wg[0],wg[1],wg[2]
        #shape = calc_size(self.mat.shape, self.wg)
        gpu_keypoints = pyopencl.array.to_device(queue, keypoints_o)
        #NOTE: for the following line, use pyopencl.array.empty instead of pyopencl.array.zeros if the keypoints are compacted
        gpu_descriptors = pyopencl.array.zeros(queue, (keypoints_end - keypoints_start, 128), dtype=numpy.uint8, order="C")
        gpu_grad = pyopencl.array.to_device(queue, grad)
        gpu_ori = pyopencl.array.to_device(queue, ori)

        keypoints_start, keypoints_end = numpy.int32(keypoints_start), numpy.int32(keypoints_end)
        grad_height, grad_width = numpy.int32(grad.shape)

        t0 = time.time()
        k1 = self.program.descriptor(queue, shape, wg,
            gpu_keypoints.data, gpu_descriptors.data, gpu_grad.data, gpu_ori.data, numpy.int32(octsize),
            keypoints_start, keypoints_end, grad_width, grad_height)
        res = gpu_descriptors.get()
        t1 = time.time()

        ref = my_descriptor(keypoints_o, grad, ori, octsize, keypoints_start, keypoints_end)
        t2 = time.time()
        
        PRINT_KEYPOINTS=True
        if (PRINT_KEYPOINTS):
#            print res[0:30,0:15]#keypoints_end-keypoints_start,0:15]
            print res[1,:]
            print ""
#            print ref[0:30,0:15]#[0:keypoints_end-keypoints_start,0:15]
            print ref[1,:]
            print res[1,:].sum(),ref[1,:].sum()
#            print res[1,:].sum(), ref[1,:].sum()
            #print keypoints_before_orientation[0:33]
        
#        sort to compare added keypoints
        delta = (res-ref).max()
        self.assert_(delta == 0, "delta=%s" % (delta)) #integers
        logger.info("delta=%s" % delta)

        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Descriptors computation took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
            
            
            
            

def test_suite_keypoints():
    testSuite = unittest.TestSuite()
#    testSuite.addTest(test_keypoints("test_orientation"))
    testSuite.addTest(test_keypoints("test_descriptor"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_keypoints()
    runner = unittest.TextTestRunner()
    if not runner.run(mysuite).wasSuccessful():
        sys.exit(1)









































