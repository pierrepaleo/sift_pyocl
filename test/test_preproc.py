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


def normalize(img, max_out=255):
    """
    Numpy implementation of the normalization
    """
    fimg = img.astype("float32")
    img_max = fimg.max()
    img_min = fimg.min()
    return max_out * (fimg - img_min) / (img_max - img_min)

def shrink(img, xs, ys):
    return img[0::ys, 0::xs]
def shrink_cython(img, xs, ys):
    try:
        import feature
    except:
        return img[0::ys, 0::xs]
    else:
        return feature.shrink(img, xs)


def binning(input_img, binsize):
    """
    @param input_img: input ndarray
    @param binsize: int or 2-tuple representing the size of the binning
    @return: binned input ndarray
    """
    inputSize = input_img.shape
    outputSize = []
    assert(len(inputSize) == 2)
    if isinstance(binsize, int):
        binsize = (binsize, binsize)
    for i, j in zip(inputSize, binsize):
        assert(i % j == 0)
        outputSize.append(i // j)

    if numpy.array(binsize).prod() < 50:
        out = numpy.zeros(tuple(outputSize))
        for i in xrange(binsize[0]):
            for j in xrange(binsize[1]):
                out += input_img[i::binsize[0], j::binsize[1]]
    else:
        temp = input_img.copy()
        temp.shape = (outputSize[0], binsize[0], outputSize[1], binsize[1])
        out = temp.sum(axis=3).sum(axis=1)
    return out

class test_preproc(unittest.TestCase):
    def setUp(self):
        self.input = scipy.misc.lena()
        self.gpudata = pyopencl.array.empty(queue, self.input.shape, dtype=numpy.float32, order="C")
        kernel_path = os.path.join(os.path.dirname(os.path.abspath(sift.__file__)), "preprocess.cl")
        kernel_src = open(kernel_path).read()
        self.program = pyopencl.Program(ctx, kernel_src).build()
        self.IMAGE_W = numpy.int32(self.input.shape[-1])
        self.IMAGE_H = numpy.int32(self.input.shape[0])
        self.wg = (2, 256)
        self.shape = calc_size(self.input.shape, self.wg)
        self.binning = (4, 2) # Nota if wg < ouptup size weired results are expected !
        self.binning = (2, 2)
        self.twofivefive = pyopencl.array.to_device(queue, numpy.array([255], numpy.float32))

    def tearDown(self):
        self.input = None
        self.program = None

    def test_uint8(self):
        """
        tests the uint8 kernel
        """
        lint = self.input.astype(numpy.uint8)
        t0 = time.time()
        au8 = pyopencl.array.to_device(queue, lint)
        k1 = self.program.u8_to_float(queue, self.shape, self.wg, au8.data, self.gpudata.data, self.IMAGE_W, self.IMAGE_H)
        min_data = pyopencl.array.min(self.gpudata, queue)
        max_data = pyopencl.array.max(self.gpudata, queue)
        k2 = self.program.normalizes(queue, self.shape, self.wg, self.gpudata.data, min_data.data, max_data.data, self.twofivefive.data, self.IMAGE_W, self.IMAGE_H)
#        k2 = self.program.normalizes(queue, self.shape, self.wg, self.gpudata.data, min_data, max_data, self.twofivefive.data)
        res = self.gpudata.get()
        t1 = time.time()
        ref = normalize(lint)
        t2 = time.time()
        delta = abs(ref - res).max()
        self.assert_(delta < 1e-4, "delta=%s" % delta)
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
        k1 = self.program.u16_to_float(queue, self.shape, self.wg, au8.data, self.gpudata.data, self.IMAGE_W, self.IMAGE_H)
        min_data = pyopencl.array.min(self.gpudata, queue)
        max_data = pyopencl.array.max(self.gpudata, queue)
        k2 = self.program.normalizes(queue, self.shape, self.wg, self.gpudata.data, min_data.data, max_data.data, self.twofivefive.data, self.IMAGE_W, self.IMAGE_H)
        k2.wait()
        res = self.gpudata.get()
        t1 = time.time()
        ref = normalize(lint)
        t2 = time.time()
        delta = abs(ref - res).max()
        self.assert_(delta < 1e-4, "delta=%s" % delta)
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
        k1 = self.program.s32_to_float(queue, self.shape, self.wg, au8.data, self.gpudata.data, self.IMAGE_W, self.IMAGE_H)
        min_data = pyopencl.array.min(self.gpudata, queue)
        max_data = pyopencl.array.max(self.gpudata, queue)
        k2 = self.program.normalizes(queue, self.shape, self.wg, self.gpudata.data, min_data.data, max_data.data, self.twofivefive.data, self.IMAGE_W, self.IMAGE_H)
        res = self.gpudata.get()
        t1 = time.time()
        ref = normalize(lint)
        t2 = time.time()
        delta = abs(ref - res).max()
        self.assert_(delta < 1e-4, "delta=%s" % delta)
        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("conversion int32->float took %.3fms and normalization took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start),
                                                                                             1e-6 * (k2.profile.end - k2.profile.start)))

    def test_int64(self):
        """
        tests the int64 kernel
        """
        lint = self.input.astype(numpy.int64)
        t0 = time.time()
        au8 = pyopencl.array.to_device(queue, lint)
        k1 = self.program.s64_to_float(queue, self.shape, self.wg, au8.data, self.gpudata.data, self.IMAGE_W, self.IMAGE_H)
        min_data = pyopencl.array.min(self.gpudata, queue)
        max_data = pyopencl.array.max(self.gpudata, queue)
        k2 = self.program.normalizes(queue, self.shape, self.wg, self.gpudata.data, min_data.data, max_data.data, self.twofivefive.data, self.IMAGE_W, self.IMAGE_H)
        res = self.gpudata.get()
        t1 = time.time()
        ref = normalize(lint)
        t2 = time.time()
        delta = abs(ref - res).max()
        self.assert_(delta < 1e-4, "delta=%s" % delta)
        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("conversion int64->float took %.3fms and normalization took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start),
                                                                                             1e-6 * (k2.profile.end - k2.profile.start)))

    def test_shrink(self):
        """
        Test shrinking kernel
        """
        lint = self.input.astype(numpy.float32)
        out_shape = tuple((i // j) for i, j in zip(self.input.shape, self.binning))
        t0 = time.time()
        inp_gpu = pyopencl.array.to_device(queue, lint)
        out_gpu = pyopencl.array.empty(queue, out_shape, dtype=numpy.float32, order="C")
        k1 = self.program.shrink(queue, calc_size(out_shape, self.wg), self.wg, inp_gpu.data, out_gpu.data,
                                 numpy.int32(self.binning[1]), numpy.int32(self.binning[0]), numpy.int32(out_shape[1]), numpy.int32(out_shape[0]))
        res = out_gpu.get()
        t1 = time.time()
        ref = shrink_cython(lint, xs=self.binning[1], ys=self.binning[0])
        t2 = time.time()
        delta = abs(ref - res).max()
        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Shrinking  took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
            fig = pylab.figure()
            fig.suptitle('Shrinking by %s,%s' % self.binning)
            sp1 = fig.add_subplot(221)
            sp1.imshow(lint, interpolation="nearest")
            sp1.set_title("Input")
            sp2 = fig.add_subplot(222)
            sp2.imshow(ref, interpolation="nearest")
            sp2.set_title("Reference")
            sp3 = fig.add_subplot(223)
            sp3.imshow(ref - res, interpolation="nearest")
            sp3.set_title("Delta= %s" % delta)
            sp4 = fig.add_subplot(224)
            sp4.imshow(res, interpolation="nearest")
            sp4.set_title("GPU")
            fig.show()
            raw_input("enter")
        self.assert_(delta < 1e-6, "delta=%s" % delta)


    def test_bin(self):
        """
        Test binning kernel
        """
        lint = numpy.ascontiguousarray(self.input, numpy.float32)

        out_shape = tuple((i // j) for i, j in zip(self.input.shape, self.binning))
        t0 = time.time()
        inp_gpu = pyopencl.array.to_device(queue, lint)
        out_gpu = pyopencl.array.empty(queue, out_shape, dtype=numpy.float32, order="C")
        k1 = self.program.bin(queue, calc_size(out_shape, self.wg), self.wg, inp_gpu.data, out_gpu.data,
                                 numpy.int32(self.binning[1]), numpy.int32(self.binning[0]),
                                 numpy.int32(lint.shape[1]), numpy.int32(lint.shape[0]),
                                 numpy.int32(out_shape[1]), numpy.int32(out_shape[0]))
        res = out_gpu.get()
        t1 = time.time()
        ref = binning(lint, self.binning) / self.binning[0] / self.binning[1]
        t2 = time.time()
        delta = abs(ref - res).max()
        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Binning took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
            fig = pylab.figure()
            fig.suptitle('Binning by %s,%s' % self.binning)
            sp1 = fig.add_subplot(221)
            sp1.imshow(lint, interpolation="nearest")
            sp1.set_title("Input")
            sp2 = fig.add_subplot(222)
            sp2.imshow(ref, interpolation="nearest")
            sp2.set_title("Reference")
            sp3 = fig.add_subplot(223)
            sp3.imshow(ref - res, interpolation="nearest")
            sp3.set_title("Delta= %s" % delta)
            sp4 = fig.add_subplot(224)
            sp4.imshow(res, interpolation="nearest")
            sp4.set_title("GPU")
            fig.show()
            raw_input("enter")
        self.assert_(delta < 1e-6, "delta=%s" % delta)

    def test_max_min(self):
        """
        Test global_max_min kernel
        """
        lint = numpy.random.random(self.input.shape).astype(numpy.float32)
        inp_gpu = pyopencl.array.to_device(queue, lint)
        size = lint.size
        blocksize = 16
        min_gpu = pyopencl.array.empty(queue, (blocksize,), dtype=numpy.float32, order="C")
        max_gpu = pyopencl.array.empty(queue, (blocksize,), dtype=numpy.float32, order="C")
        t = time.time()
        nmin = lint.min()
        nmax = lint.max()
        t0 = time.time()
        k1 = self.program.global_max_min(queue, (size,), (blocksize,), inp_gpu.data, max_gpu.data, min_gpu.data,
                                        numpy.int32(lint.size))
        min_res = min_gpu.get()
        max_res = max_gpu.get()
        print min_res
        print max_res

#        k2 = self.program.global_max_min(queue, (blocksize,), (blocksize,), max_gpu.data, max_gpu.data, min_gpu.data,
#                                        numpy.int32(blocksize))
#        k3 = self.program.global_max_min(queue, (blocksize,), (blocksize,), min_gpu.data, max_gpu.data, min_gpu.data,
#                                        numpy.int32(blocksize))
#        min_res = min_gpu.get()
#        max_res = max_gpu.get()
        t1 = time.time()
        min_pyocl = pyopencl.array.min(inp_gpu, queue).get()
        max_pyocl = pyopencl.array.max(inp_gpu, queue).get()
        t2 = time.time()
        print min_res
        print max_res
        max_res = max_res.max()
        min_res = min_res.min()
        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms, pyopencl: %.3fms." % (1000.0 * (t0 - t), 1000.0 * (t1 - t0), 1000.0 * (t2 - t1)))
            logger.info("reduction took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
        print nmin, min_res, min_pyocl
        print nmax, max_res, max_pyocl
        self.assertEqual(nmin, min_res, "min: numpy vs ours")
        self.assertEqual(nmax, max_res, "max: numpy vs ours")
        self.assertEqual(nmin, min_pyocl, "min: numpy vs pyopencl")
        self.assertEqual(nmax, max_pyocl, "max: numpy vs pyopencl")
        self.assertEqual(min_pyocl, min_res, "min: ours vs pyopencl")
        self.assertEqual(max_pyocl, max_res, "max: ours vs pyopencl")


def test_suite_preproc():
    testSuite = unittest.TestSuite()
#    testSuite.addTest(test_preproc("test_uint8"))
#    testSuite.addTest(test_preproc("test_uint16"))
#    testSuite.addTest(test_preproc("test_int32"))
#    testSuite.addTest(test_preproc("test_int64"))
#    testSuite.addTest(test_preproc("test_shrink"))
#    testSuite.addTest(test_preproc("test_bin"))
    testSuite.addTest(test_preproc("test_max_min"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_preproc()
    runner = unittest.TextTestRunner()
    if not runner.run(mysuite).wasSuccessful():
        sys.exit(1)

