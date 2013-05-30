#!/usr/bin/env python
#-*- coding: utf8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/kif/sift_pyocl
#

"""
Test suite for all preprocessing kernels.
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
import scipy, scipy.misc
import sys
import unittest
from utilstest import UtilsTest, getLogger
import sift
from sift.opencl import ocl
logger = getLogger(__file__)
ctx = ocl.create_context("GPU")
if logger.getEffectiveLevel() <= logging.INFO:
    PROFILE = True
    queue = pyopencl.CommandQueue(ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
else:
    PROFILE = False
    queue = pyopencl.CommandQueue(ctx)

print "working on %s" % ctx.devices[0].name

def normalize(img, max_out=255):
    """
    Numpy implementation of the normalization
    """
    fimg = img.astype("float32")
    img_max = fimg.max()
    img_min = fimg.min()
    return max_out * (fimg - img_min) / (img_max - img_min)

class test_preproc(unittest.TestCase):
    def setUp(self):
        self.input = scipy.misc.lena()
        self.gpudata = pyopencl.array.empty(queue, self.input.shape, dtype=numpy.float32, order="C")
#        self.maxout = pyopencl.array.to_device(queue, numpy.array([255.0], dtype=numpy.float32))
        kernel_path = os.path.join(os.path.dirname(os.path.abspath(sift.__file__)), "preprocess.cl")
        kernel_src = open(kernel_path).read()
        compile_options = "-D NIMAGE=%i" % self.input.size
        logger.info("Compiling file %s with options %s" % (kernel_path, compile_options))
        self.program = pyopencl.Program(ctx, kernel_src).build(options=compile_options)
        self.wg = (1024,)

    def test_uint8(self):
        """
        tests the uint8 kernel
        """
        lint = self.input.astype(numpy.uint8)
        t0 = time.time()
        au8 = pyopencl.array.to_device(queue, lint)
        k1 = self.program.u8_to_float(queue, (self.input.size,), self.wg, au8.data, self.gpudata.data)
        min_data = pyopencl.array.min(self.gpudata, queue).get()
        max_data = pyopencl.array.max(self.gpudata, queue).get()
        k2 = self.program.normalizes(queue, (self.input.size,), self.wg, self.gpudata.data, numpy.float32(min_data), numpy.float32(max_data), numpy.float32(255))
#        k2 = self.program.normalizes(queue, (self.input.size,), self.wg, self.gpudata.data, min_data, max_data, numpy.float32(255))
        res = self.gpudata.get()
        t1 = time.time()
        ref = normalize(lint)
        t2 = time.time()
        delta = abs(ref - res).max()
        self.assert_(delta < 1e-4, "delta=%s")
        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("conversion uint8->float took %.3fms and normalization took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start),
                                                                                             1e-6 * (k2.profile.end - k2.profile.start)))
    def test_uint16(self):
        """
        tests the uint16 kernel
        """
        lint = self.input.astype(numpy.uint16)
        t0 = time.time()
        au8 = pyopencl.array.to_device(queue, lint)
        k1 = self.program.u16_to_float(queue, (self.input.size,), self.wg, au8.data, self.gpudata.data)
        min_data = pyopencl.array.min(self.gpudata, queue).get()
        max_data = pyopencl.array.max(self.gpudata, queue).get()
        k2 = self.program.normalizes(queue, (self.input.size,), self.wg, self.gpudata.data, numpy.float32(min_data), numpy.float32(max_data), numpy.float32(255))
        res = self.gpudata.get()
        t1 = time.time()
        ref = normalize(lint)
        t2 = time.time()
        delta = abs(ref - res).max()
        self.assert_(delta < 1e-4, "delta=%s")
        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("conversion uint16->float took %.3fms and normalization took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start),
                                                                                             1e-6 * (k2.profile.end - k2.profile.start)))
    def test_int32(self):
        """
        tests the int32 kernel
        """
        lint = self.input.astype(numpy.int32)
        t0 = time.time()
        au8 = pyopencl.array.to_device(queue, lint)
        k1 = self.program.s32_to_float(queue, (self.input.size,), self.wg, au8.data, self.gpudata.data)
        min_data = pyopencl.array.min(self.gpudata, queue).get()
        max_data = pyopencl.array.max(self.gpudata, queue).get()
        k2 = self.program.normalizes(queue, (self.input.size,), self.wg, self.gpudata.data, numpy.float32(min_data), numpy.float32(max_data), numpy.float32(255))
        res = self.gpudata.get()
        t1 = time.time()
        ref = normalize(lint)
        t2 = time.time()
        delta = abs(ref - res).max()
        self.assert_(delta < 1e-4, "delta=%s")
        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("conversion int32->float took %.3fms and normalization took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start),
                                                                                             1e-6 * (k2.profile.end - k2.profile.start)))


def test_suite_preproc():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_preproc("test_uint8"))
    testSuite.addTest(test_preproc("test_uint16"))
    testSuite.addTest(test_preproc("test_int32"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_preproc()
    runner = unittest.TextTestRunner()
    if not runner.run(mysuite).wasSuccessful():
        sys.exit(1)

