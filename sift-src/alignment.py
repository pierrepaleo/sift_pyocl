#!/usr/bin/env python
# -*- coding: utf8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/kif/sift_pyocl
#

"""
Contains classes for image alignment on a reference images.
"""

from __future__ import division, print_function, with_statement

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "BSD"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "2013-07-24"
__status__ = "beta"
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
import os, gc
from threading import Semaphore
import numpy
import pyopencl, pyopencl.array
from .param import par
from .opencl import ocl
from .utils import calc_size, kernel_size, sizeof, matching_correction
import logging
logger = logging.getLogger("sift.alignment")
from pyopencl import mem_flags as MF
from . import MatchPlan, SiftPlan

class LinearAlign(object):
    """
    Align images on a reference image based on an Afine transformation (bi-linear + offset)
    """
    kernel_file = "transform"

    def __init__(self, image, devicetype="CPU", profile=False, device=None, max_workgroup_size=128, roi=None, extra=0, context=None):
        """
        Constructor of the class

        @param image: reference image on which other image should be aligned
        @param devicetype: Kind of prefered devce
        @param profile:collect profiling information ?
        @param device: 2-tuple of integer. see clinfo
        @param max_workgroup_size: seet to 1 for macOSX
        @param roi: Region of interest: to be implemented
        @param extra: extra space around the image, can be an integer, or a 2 tuple in YX convention: TODO!
        """
        self.profile = bool(profile)
        self.events = []
        self.program = None
        self.ref = numpy.ascontiguousarray(image, numpy.float32)
        self.buffers = {}
        self.shape = image.shape
        if len(self.shape) == 3:
            self.RGB = True
            self.shape = self.shape[:2]
        elif len(self.shape) == 2:
            self.RGB = False
        else:
            raise RuntimeError("Unable to process image of shape %s" % (tuple(self.shape,)))
        if "__len__" not in dir(extra):
            self.extra = (int(extra), int(extra))
        else:
            self.extra = extra[:2]
        self.outshape = tuple(i + 2 * j for i, j in zip(self.shape, self.extra))
        if self.RGB:
            self.wg = (4, 8, 4)
        else:
            self.wg = (8, 4)
        if context:
            self.ctx = context
            device_name = self.ctx.devices[0].name.strip()
            platform_name = self.ctx.devices[0].platform.name.strip()
            platform = ocl.get_platform(platform_name)
            device = platform.get_device(device_name)
            self.device = platform.id, device.id
        else:
            if device is None:
                self.device = ocl.select_device(type=devicetype, best=True)
            else:
                self.device = device
            self.ctx = pyopencl.Context(devices=[pyopencl.get_platforms()[self.device[0]].get_devices()[self.device[1]]])

        self.sift = SiftPlan(template=image, context=self.ctx, profile=self.profile, max_workgroup_size=max_workgroup_size)
        self.ref_kp = self.sift.keypoints(image)
        self.match = MatchPlan(context=self.ctx, profile=self.profile, max_workgroup_size=max_workgroup_size, roi=roi)
#        Allocate reference keypoints on the GPU within match context:
        self.buffers["ref_kp_gpu"] = pyopencl.array.to_device(self.match.queue, self.ref_kp)
        #TODO optimize match so that the keypoint2 can be optional
        self.fill_value = 0
#        print self.ctx.devices[0]
        if self.profile:
            self.queue = pyopencl.CommandQueue(self.ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
        else:
            self.queue = pyopencl.CommandQueue(self.ctx)
        self._compile_kernels()
        self._allocate_buffers()
        self.sem = Semaphore()

    def __del__(self):
        """
        Destructor: release all buffers
        """
        self._free_kernels()
        self._free_buffers()
        self.queue = None
        self.ctx = None
        gc.collect()

    def _allocate_buffers(self):
        """
        All buffers are allocated here
        """
        if self.RGB:
            self.buffers["input"] = pyopencl.array.empty(self.queue, shape=self.shape + (3,), dtype=numpy.uint8)
            self.buffers["output"] = pyopencl.array.empty(self.queue, shape=self.outshape + (3,), dtype=numpy.uint8)
        else:
            self.buffers["input"] = pyopencl.array.empty(self.queue, shape=self.shape, dtype=numpy.float32)
            self.buffers["output"] = pyopencl.array.empty(self.queue, shape=self.outshape, dtype=numpy.float32)
        self.buffers["matrix"] = pyopencl.array.empty(self.queue, shape=(2, 2), dtype=numpy.float32)
        self.buffers["offset"] = pyopencl.array.empty(self.queue, shape=(1, 2), dtype=numpy.float32)
    def _free_buffers(self):
        """
        free all memory allocated on the device
        """
        for buffer_name in self.buffers:
            if self.buffers[buffer_name] is not None:
                try:
                    del self.buffers[buffer_name]
                    self.buffers[buffer_name] = None
                except pyopencl.LogicError:
                    logger.error("Error while freeing buffer %s" % buffer_name)

    def _compile_kernels(self):
        """
        Call the OpenCL compiler
        """
        kernel_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.kernel_file + ".cl")
        kernel_src = open(kernel_file).read()
        try:
            program = pyopencl.Program(self.ctx, kernel_src).build()
        except pyopencl.MemoryError as error:
            raise MemoryError(error)
        self.program = program

    def _free_kernels(self):
        """
        free all kernels
        """
        self.program = None

    def align(self, img, shift_only=False, all=False):
        """
        Align image on reference image
        
        @param img: numpy array containing the image to align to reference
        @param all: return in addition ot the image, keypoints, matching keypoints, and transformations as a dict
        @return: aligned image
        """
        logger.debug("ref_keypoints: %s" % self.ref_kp.size)
        if self.RGB:
            data = numpy.ascontiguousarray(img, numpy.uint8)
        else:
            data = numpy.ascontiguousarray(img, numpy.float32)
        with self.sem:
            cpy = pyopencl.enqueue_copy(self.queue, self.buffers["input"].data, data)
            if self.profile:self.events.append(("Copy H->D", cpy))
            cpy.wait()
            kp = self.sift.keypoints(self.buffers["input"])
            logger.debug("mod image keypoints: %s" % kp.size)
            raw_matching = self.match.match(self.buffers["ref_kp_gpu"], kp, raw_results=True)
            matching = numpy.recarray(shape=raw_matching.shape, dtype=MatchPlan.dtype_kp)
            len_match = raw_matching.shape[0]
            if len_match == 0:
                logger.warning("No matching keypoints")
                return
            matching[:, 1] = self.ref_kp[raw_matching[:, 0]]
            matching[:, 0] = kp[raw_matching[:, 1]]

            if (len_match < 3 * 6) or (shift_only): # 3 points per DOF
                logger.warning("Shift Only mode: Common keypoints: %s" % len_match)
                dx = matching[:, 1].x - matching[:, 0].x
                dy = matching[:, 1].y - matching[:, 0].y
                matrix = numpy.identity(2, dtype=numpy.float32)
                offset = numpy.array([numpy.median(dy), numpy.median(dx)], numpy.float32)
            else:
                logger.debug("Common keypoints: %s" % len_match)

                transform_matrix = matching_correction(matching)
                transform_matrix.shape = 2, 3
                matrix = numpy.ascontiguousarray(transform_matrix[:, :2], dtype=numpy.float32)
                offset = numpy.ascontiguousarray(transform_matrix[:, -1], dtype=numpy.float32)
            cpy1 = pyopencl.enqueue_copy(self.queue, self.buffers["matrix"].data, matrix)
            cpy2 = pyopencl.enqueue_copy(self.queue, self.buffers["offset"].data, offset)
            if self.profile:
                self.events += [("Copy matrix", cpy1), ("Copy offset", cpy2)]

            if self.RGB:
                shape = (4, self.shape[1], self.shape[0])
                transform = self.program.transform_RGB
            else:
                shape = self.shape[1], self.shape[0]
                transform = self.program.transform
            ev = transform(self.queue, calc_size(shape, self.wg), self.wg,
                                   self.buffers["input"].data,
                                   self.buffers["output"].data,
                                   self.buffers["matrix"].data,
                                   self.buffers["offset"].data,
                                   numpy.int32(self.shape[1]),
                                   numpy.int32(self.shape[0]),
                                   numpy.int32(self.outshape[1]),
                                   numpy.int32(self.outshape[0]),
                                   self.sift.buffers["min"].get()[0],
                                   numpy.int32(1))
            if self.profile:
                self.events += [("transform", ev)]
            result = self.buffers["output"].get()
        if all:
            return {"result":result, "keypoint":kp, "matching":matching, "offset":offset, "matrix": matrix}
        return result

    def log_profile(self):
        """
        If we are in debugging mode, prints out all timing for every single OpenCL call
        """
        t = 0.0
        orient = 0.0
        descr = 0.0
        if self.profile:
            for e in self.events:
                if "__len__" in dir(e) and len(e) >= 2:
                    et = 1e-6 * (e[1].profile.end - e[1].profile.start)
                    print("%50s:\t%.3fms" % (e[0], et))
                    t += et



