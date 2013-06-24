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
PRINT_KEYPOINTS = True


print "working on %s" % ctx.devices[0].name

'''
For Python implementation of tested functions, see "test_image_functions.py"
'''



class test_image(unittest.TestCase):
    def setUp(self):

        kernel_path = os.path.join(os.path.dirname(os.path.abspath(sift.__file__)), "image.cl")
        kernel_src = open(kernel_path).read()
        self.program = pyopencl.Program(ctx, kernel_src).build()
        self.wg = (1, 8)



    def tearDown(self):
        self.mat = None
        self.program = None






    def test_gradient(self):
        """
        tests the gradient kernel (norm and orientation)
        """

        self.width = numpy.int32(15)
        self.height = numpy.int32(14)

        self.mat = numpy.random.rand(self.height, self.width).astype(numpy.float32)
        self.gpu_mat = pyopencl.array.to_device(queue, self.mat)
        self.gpu_grad = pyopencl.array.empty(queue, self.mat.shape, dtype=numpy.float32, order="C")
        self.gpu_ori = pyopencl.array.empty(queue, self.mat.shape, dtype=numpy.float32, order="C")
        self.shape = calc_size(self.mat.shape, self.wg)

        t0 = time.time()
        k1 = self.program.compute_gradient_orientation(queue, self.shape, self.wg, self.gpu_mat.data, self.gpu_grad.data, self.gpu_ori.data, self.width, self.height)
        res_norm = self.gpu_grad.get()
        res_ori = self.gpu_ori.get()
        t1 = time.time()
        ref_norm, ref_ori = my_gradient(self.mat)
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
        #local_maxmin_setup :
        border_dist, peakthresh, EdgeThresh, EdgeThresh0, octsize, s, nb_keypoints, width, height, DOGS, g = local_maxmin_setup()
        self.s = numpy.int32(s) #1, 2, 3 ... not 4 nor 0.
        self.gpu_dogs = pyopencl.array.to_device(queue, DOGS)
        self.output = pyopencl.array.empty(queue, (nb_keypoints, 4), dtype=numpy.float32, order="C")
        self.output.fill(-1.0, queue) #memset for invalid keypoints
        self.counter = pyopencl.array.zeros(queue, (1,), dtype=numpy.int32, order="C")
        nb_keypoints = numpy.int32(nb_keypoints)
        self.shape = calc_size((DOGS.shape[1], DOGS.shape[0] * DOGS.shape[2]), self.wg) #it's a 3D vector !!

        t0 = time.time()
        k1 = self.program.local_maxmin(queue, self.shape, self.wg,
        	self.gpu_dogs.data, self.output.data,
       		border_dist, peakthresh, octsize, EdgeThresh0, EdgeThresh,
       		self.counter.data, nb_keypoints, self.s, width, height)

        res = self.output.get()
        self.keypoints1 = self.output #for further use
        self.actual_nb_keypoints = self.counter.get()[0] #for further use

        t1 = time.time()
        ref, actual_nb_keypoints2 = my_local_maxmin(DOGS, peakthresh, border_dist, octsize,
        	EdgeThresh0, EdgeThresh, nb_keypoints, self.s, width, height)
        t2 = time.time()

        #we have to sort the arrays, for peaks orders is unknown for GPU
        res_peaks = res[(res[:, 0].argsort(axis=0)), 0]
        ref_peaks = ref[(ref[:, 0].argsort(axis=0)), 0]
        res_r = res[(res[:, 1].argsort(axis=0)), 1]
        ref_r = ref[(ref[:, 1].argsort(axis=0)), 1]
        res_c = res[(res[:, 2].argsort(axis=0)), 2]
        ref_c = ref[(ref[:, 2].argsort(axis=0)), 2]
        #res_s = res[(res[:,3].argsort(axis=0)),3]
        #ref_s = ref[(ref[:,3].argsort(axis=0)),3]
        delta_peaks = abs(ref_peaks - res_peaks).max()
        delta_r = abs(ref_r - res_r).max()
        delta_c = abs(ref_c - res_c).max()

        if (PRINT_KEYPOINTS):
            print("keypoints after 2 steps of refinement: (s= %s, octsize=%s) %s" % (self.s, octsize, self.actual_nb_keypoints))
            #print("For ref: %s" %(ref_peaks[ref_peaks!=-1].shape))
            print res[0:self.actual_nb_keypoints]#[0:74]
            #print ref[0:32]

        self.assert_(delta_peaks < 1e-4, "delta_peaks=%s" % (delta_peaks))
        self.assert_(delta_r < 1e-4, "delta_r=%s" % (delta_r))
        self.assert_(delta_c < 1e-4, "delta_c=%s" % (delta_c))
        logger.info("delta_peaks=%s" % delta_peaks)
        logger.info("delta_r=%s" % delta_r)
        logger.info("delta_c=%s" % delta_c)


        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Local extrema search took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))





    def test_interpolation(self):
        """
        tests the keypoints interpolation kernel
        Requires the following: "self.keypoints1", "self.actual_nb_keypoints", 	"self.gpu_dog_prev", self.gpu_dog", 			"self.gpu_dog_next", "self.s", "self.width", "self.height", "self.peakthresh"
        """

        #interpolation_setup :
        border_dist, peakthresh, EdgeThresh, EdgeThresh0, octsize, nb_keypoints, actual_nb_keypoints, width, height, DOGS, s, keypoints_prev, blur = interpolation_setup()

        # actual_nb_keypoints is the number of keypoints returned by "local_maxmin".
        #After the interpolation, it will be reduced, but we can still use it as a boundary.
        shape = calc_size(keypoints_prev.shape, self.wg)
        gpu_dogs = pyopencl.array.to_device(queue, DOGS)
        gpu_keypoints1 = pyopencl.array.to_device(queue, keypoints_prev)
        #actual_nb_keypoints = numpy.int32(len((keypoints_prev[:,0])[keypoints_prev[:,1] != -1]))
        start_keypoints = numpy.int32(0)
        actual_nb_keypoints = numpy.int32(actual_nb_keypoints)
        InitSigma = numpy.float32(1.6) #warning: it must be the same in my_keypoints_interpolation
        t0 = time.time()
        k1 = self.program.interp_keypoint(queue, shape, self.wg,
        	gpu_dogs.data, gpu_keypoints1.data, start_keypoints, actual_nb_keypoints,
        	peakthresh, InitSigma, width, height)
        res = gpu_keypoints1.get()

        t1 = time.time()
        ref = numpy.copy(keypoints_prev) #important here
        for i, k in enumerate(ref[:nb_keypoints, :]):
            ref[i] = my_interp_keypoint(DOGS, s, k[1], k[2], 5, peakthresh, width, height)

        t2 = time.time()


        #we have to compare keypoints different from (-1,-1,-1,-1)
        res2 = res[res[:, 1] != -1]
        ref2 = ref[ref[:, 1] != -1]


        if (PRINT_KEYPOINTS):
            print("[s=%s]Keypoints before interpolation: %s" % (s, actual_nb_keypoints))
            #print keypoints_prev[0:10,:]
            print("[s=%s]Keypoints after interpolation : %s" % (s, res2.shape[0]))
            print res[0:actual_nb_keypoints]#[0:10,:]
            #print("Ref:")
            #print ref[0:32,:]


        delta = abs(ref2 - res2).max()
        self.assert_(delta < 1e-4, "delta=%s" % (delta))
        logger.info("delta=%s" % delta)

        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Keypoints interpolation took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))



    def test_orientation(self):
        '''
        #tests keypoints orientation assignment kernel
        '''

        #orientation_setup :
        keypoints, nb_keypoints, updated_nb_keypoints, grad, ori, octsize = orientation_setup()
        #keypoints is a compacted vector of keypoints #not anymore
        keypoints_before_orientation = numpy.copy(keypoints) #important here
        wg = max(self.wg),
        shape = calc_size((keypoints.shape[0],), wg)
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

        #print keypoints_before_orientation[0:33]
        if (PRINT_KEYPOINTS):
            print("Keypoints after orientation assignment :")
            print res[0:actual_nb_keypoints]#[0:10]
            #print " "
            #print ref[0:7]

        print("Total keypoints for kernel : %s -- For Python : %s \t [octsize = %s]" % (cnt, updated_nb_keypoints, octsize))

        #sort to compare added keypoints
        d1, d2, d3, d4 = keypoints_compare(ref, res)
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
        keypoints_o, nb_keypoints, actual_nb_keypoints, grad, ori = descriptor_setup()
        #keypoints should be a compacted vector of keypoints
        keypoints_start, keypoints_end = 0, 80 #actual_nb_keypoints
        #keypoints_start, keypoints_end = 20, 30
        keypoints = keypoints_o[keypoints_start:keypoints_end]
        print("Working on keypoints : [%s,%s]" % (keypoints_start, keypoints_end))
        wg = max(self.wg),
        shape = calc_size((keypoints_o.shape[0],), wg)
        gpu_keypoints = pyopencl.array.to_device(queue, keypoints_o)
        gpu_descriptors = pyopencl.array.empty(queue, (keypoints_end - keypoints_start + 1, 128), dtype=numpy.uint8, order="C")
        gpu_grad = pyopencl.array.to_device(queue, grad)
        gpu_ori = pyopencl.array.to_device(queue, ori)

        local_size = (keypoints_end - keypoints_start + 1) * 128 * 4
        local_mem = pyopencl.LocalMemory(local_size)

        keypoints_start, keypoints_end = numpy.int32(keypoints_start), numpy.int32(keypoints_end)
        grad_height, grad_width = numpy.int32(grad.shape)

        t0 = time.time()
        k1 = self.program.descriptor(queue, shape, wg,
            gpu_keypoints.data, gpu_descriptors.data, local_mem, gpu_grad.data, gpu_ori.data,
            keypoints_start, keypoints_end, grad_width, grad_height)
        res = gpu_descriptors.get()
        t1 = time.time()

        ref = my_descriptor(keypoints_o, grad, ori, keypoints_start, keypoints_end)
        t2 = time.time()

        print res[0:30,0:15]#keypoints_end-keypoints_start,0:15]
        #print ""
        #print ref[0:keypoints_end-keypoints_start,0:15]


        #print keypoints_before_orientation[0:33]
        #if (PRINT_KEYPOINTS):


#        TODO
#        #sort to compare added keypoints
#        d1,d2,d3,d4 = keypoints_compare(ref,res)
#        self.assert_(d1 < 1e-4, "delta_cols=%s" % (d1))
#        self.assert_(d2 < 1e-4, "delta_rows=%s" % (d2))
#        self.assert_(d3 < 1e-4, "delta_sigma=%s" % (d3))
#        self.assert_(d4 < 1e-4, "delta_angle=%s" % (d4))
#        logger.info("delta_cols=%s" % d1)
#        logger.info("delta_rows=%s" % d2)
#        logger.info("delta_sigma=%s" % d3)
#        logger.info("delta_angle=%s" % d4)


        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Descriptors computation took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
















































def test_suite_image():
    testSuite = unittest.TestSuite()
    #testSuite.addTest(test_image("test_gradient"))
    #testSuite.addTest(test_image("test_local_maxmin"))
    #testSuite.addTest(test_image("test_interpolation"))
    #testSuite.addTest(test_image("test_orientation"))
    testSuite.addTest(test_image("test_descriptor"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_image()
    runner = unittest.TextTestRunner()
    if not runner.run(mysuite).wasSuccessful():
        sys.exit(1)

