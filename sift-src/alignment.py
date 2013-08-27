#!/usr/bin/env python
# -*- coding: utf8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/kif/sift_pyocl
#

"""
Contains classes for image alignment on a reference images. 
"""

from __future__ import division

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
import numpy
import pyopencl, pyopencl.array
from .param import par
from .opencl import ocl
from .utils import calc_size, kernel_size, sizeof, matching_correction
import logging
logger = logging.getLogger("sift.alignment")
from pyopencl import mem_flags as MF
#from scipy.optimize import leastsq
from . import MatchPlan, SiftPlan

class LinearAlign(object):
    """
    Align images on a reference image
    """
    def __init__(self, image, devicetype="CPU", profile=False, device=None, max_workgroup_size=128, roi=None, extra=None):
        """
        
        @param extra: extra space around the image, can be an integer, or a 2 tuple in YX convension
        """
        self.ref = image
        self.sift = SiftPlan(template = image, devicetype=devicetype, profile=profile, device=device, max_workgroup_size=max_workgroup_size)
        self.kp = self.sift.keypoints(image)
        self.match =  MatchPlan(devicetype=devicetype, profile=profile, device=device, max_workgroup_size=max_workgroup_size, roi=roi)
        #TODO optimize match so that the keypoint2 can be optional 
    def align(self, img, extra=None):
        """
        Align 
        """
        kp = self.sift.keypoints(img)
        matching = self.match(kp, self.kp)
        #TODO optimize match so that the keypoint2 can be optional
        
        transform_matrix = matching_correction(matching)
        #TODO : call transform kernel to correct image
        
        newimg = numpy.empty_like(img)
        return newimg
        
        
        
        
        
        
        
        
