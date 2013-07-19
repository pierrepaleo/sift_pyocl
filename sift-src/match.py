#!/usr/bin/env python
# -*- coding: utf8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/kif/sift_pyocl
#

"""
Contains a class for creating a matching plan, allocating arrays, compiling kernels and other things like that
"""

from __future__ import division

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "BSD"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "2013-07-16"
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
import time, math, os, logging, sys
import gc
import numpy
import pyopencl, pyopencl.array
from .param import par
from .opencl import ocl
from .utils import calc_size, kernel_size, sizeof
logger = logging.getLogger("sift.match")
from pyopencl import mem_flags as MF

class MatchPlan(object):
    """
    Plan to compare sets of SIFT keypoint

    siftp = sift.MatchPlan(devicetype="GPU")
    kp = siftp.match(kp1,kp2)

    kp is a nx132 array. the second dimension is composed of x,y, scale and angle as well as 128 floats describing the keypoint

    """
    kernels = {"matching":1024,
               "memset":128, }
#               "keypoints":128}
    dtype_kp = numpy.dtype([('x', numpy.float32),
                                ('y', numpy.float32),
                                ('scale', numpy.float32),
                                ('angle', numpy.float32),
                                ('desc', (numpy.uint8, 128))
                                ])

    def __init__(self, size=16384, devicetype="CPU", profile=False, device=None, max_workgroup_size=128):
        """
        Contructor of the class
        """
        self.profile = bool(profile)
        self.max_workgroup_size = max_workgroup_size
        self.events = []
        self.kpsize = size
        self.buffers = {}
        self.programs = {}
        self.memory = None
        self.octave_max = None
        self.red_size = None
        if device is None:
            self.device = ocl.select_device(type=devicetype, memory=self.memory, best=True)
        else:
            self.device = device
        self.ctx = pyopencl.Context(devices=[pyopencl.get_platforms()[self.device[0]].get_devices()[self.device[1]]])
        if profile:
            self.queue = pyopencl.CommandQueue(self.ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
        else:
            self.queue = pyopencl.CommandQueue(self.ctx)
#        self._calc_workgroups()
        self._compile_kernels()
        self._allocate_buffers()
        self.debug = []

        self.devicetype = ocl.platforms[self.device[0]].devices[self.device[1]].type
        if (self.devicetype == "CPU"):
            self.USE_CPU = True
        else:
            self.USE_CPU = False

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
        self.buffers[ "Kp_1" ] = pyopencl.array.empty(self.queue, (self.kpsize, 4), dtype=numpy.float32)
        self.buffers[ "Kp_2" ] = pyopencl.array.empty(self.queue, (self.kpsize, 4), dtype=numpy.float32)
        self.buffers[ "match" ] = pyopencl.array.empty(self.queue, (self.kpsize, 2), dtype=numpy.int32)
        # self.buffers["cnt" ] = pyopencl.array.empty(self.queue, 1, dtype=numpy.int32)

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
        for kernel in self.kernels:
            kernel_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), kernel + ".cl")
            kernel_src = open(kernel_file).read()
            wg_size = min(self.max_workgroup_size, self.kernels[kernel])
            try:
                program = pyopencl.Program(self.ctx, kernel_src).build('-D WORKGROUP_SIZE=%s' % wg_size)
            except pyopencl.MemoryError as error:
                raise MemoryError(error)
            except pyopencl.RuntimeError as error:
                if kernel == "keypoints":
                    logger.warning("Failed compiling kernel '%s' with workgroup size %s: %s: use low_end alternative", kernel, wg_size, error)
                    self.LOW_END = True
                else:
                    logger.error("Failed compiling kernel '%s' with workgroup size %s: %s", kernel, wg_size, error)
                    raise error
            self.programs[kernel] = program

    def _free_kernels(self):
        """
        free all kernels
        """
        self.programs = {}
