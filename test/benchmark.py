#!/usr/bin/python
# -*- coding: utf-8 -*
"""
Small demonstration for sift using pyOpenCL
"""
from __future__ import division

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "2013-07-15"
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

import sys, os, pyopencl, time
from math import sin, cos
import utilstest
logger = utilstest.getLogger(__file__)
import sift_pyocl as sift
import numpy
import scipy.misc
import pylab
SHOW_FIGURES = False

try:
    import feature
except:
    logger.error("Feature is not available to compare results with C++ implementation")
    feature = None
    res = numpy.empty(0)

def cmp_kp(a, b):
    if a.scale < b.scale:
        return True
    elif a.scale > b.scale:
        return False
    else:
        if a.angle > b.angle:
            return True
        else:
            return False


class DemoSift(object):
    def __init__(self, fname, devicetype=None, device=None, context=None):


        self.image_rgb = scipy.misc.imread(fname)
        self.image_bw = 0.299 * self.image_rgb[:, :, 0] + 0.587 * self.image_rgb[:, :, 1] + 0.114 * self.image_rgb[:, :, 2]
        if feature:
            self._sift_cpp = feature.SiftAlignment()
        self._sift_ocl = sift.SiftPlan(template=self.image_rgb, device=device, devicetype=devicetype, context=context)
        self.kp_cpp = numpy.empty(0)
        self.kp_ocl = numpy.empty(0)

        if SHOW_FIGURES == True:
            self.fig = pylab.figure()
            self.sp1 = self.fig.add_subplot(1, 2, 1)
            self.im1 = self.sp1.imshow(self.image_rgb)
            self.sp1.set_title("OpenCL: %s keypoint" % self.kp_ocl.size)
            self.sp2 = self.fig.add_subplot(1, 2, 2)
            self.im2 = self.sp2.imshow(self.image_bw, cmap="gray")
            self.sp2.set_title("C++: %s keypoint" % self.kp_cpp.size)
            self.fig.show()

        self.timing_cpp = None
        self.timing_ocl = None
        self.speedups = numpy.zeros((1, 3), dtype=numpy.float32)

    def sift_cpp(self):
        print("Running SIFT using C++ code")
        t0 = time.time()
        self.kp_cpp = self._sift_cpp.sift(self.image_bw)
        t1 = time.time()
        self.timing_cpp = t1 - t0
        if SHOW_FIGURES == True:
            self.sp2.set_title("C++: %s keypoint" % self.kp_cpp.size)
            self.fig.canvas.draw()
        self.kp_cpp.sort(order=["scale", "angle", "x", "y"])
        return self.kp_cpp.size

    def sift_ocl(self):
        print("Running SIFT using OpenCL code")
        t0 = time.time()
        self.kp_ocl = self._sift_ocl.keypoints(self.image_rgb)
        t1 = time.time()
        self.timing_ocl = t1 - t0
        if SHOW_FIGURES == True:
            self.sp1.set_title("OpenCL: %s keypoint" % self.kp_ocl.size)
            self.fig.canvas.draw()
        self.kp_ocl.sort(order=["scale", "angle", "x", "y"])
        return self.kp_ocl.size

    def timings(self, kp_ocl):
        if self.kp_ocl.size > 0 and self.kp_cpp > 0:
            speedup = self.timing_cpp / self.timing_ocl
            self.speedups[0] = speedup, self.image_bw.size, kp_ocl
            print("Computing time using C++: %.3fms\t using OpenCL: %.3fms:\t Speed up: %.3f" % (1e3 * self.timing_cpp, 1e3 * self.timing_ocl, speedup))
        return self.speedups


    def show(self, max=1000):
        if max == None:
            max = sys.maxint
        if self.kp_cpp.size > max:
            print("keeping only the %i largest keypoints for display" % max)
        todo = min(self.kp_cpp.size, max)

        for i in range(self.kp_cpp.size - todo, self.kp_cpp.size):
            x = self.kp_cpp[i].x
            y = self.kp_cpp[i].y
            scale = self.kp_cpp[i].scale
            angle = self.kp_cpp[i].angle
            x0 = x + scale * cos(angle)
            y0 = y + scale * sin(angle)
            self.sp2.annotate("", xy=(x, y), xytext=(x0, y0), color="red",
                             arrowprops=dict(facecolor='red', edgecolor='red', width=1),)
            self.sp1.annotate("", xy=(x, y), xytext=(x0, y0), color="red",
                             arrowprops=dict(facecolor='red', edgecolor='red', width=1),)
        self.fig.canvas.draw()

        todo = min(self.kp_ocl.size, max)
        for i in range(self.kp_ocl.size - todo, self.kp_ocl.size):
            x = self.kp_ocl[i].x
            y = self.kp_ocl[i].y
            scale = self.kp_ocl[i].scale
            angle = self.kp_ocl[i].angle
            x0 = x + scale * cos(angle)
            y0 = y + scale * sin(angle)
            self.sp1.annotate("", xy=(x, y), xytext=(x0, y0), color="blue",
                             arrowprops=dict(facecolor='blue', edgecolor='blue', width=1),)
        self.fig.canvas.draw()

    def match(self):
        if self.kp_ocl.size > 0 and self.kp_cpp > 0:
            t0 = time.time()
            match = feature.sift_match(self.kp_ocl, self.kp_cpp)
            t1 = time.time()
            print("Number of Keypoints with OpenCL: %i, With C++: %i, Matching keypoints: %i, Processing time: %.3fs" % (self.kp_ocl.size, self.kp_cpp.size, match.shape[0], (t1 - t0)))
            self.fig.canvas.draw()
        else:
            print("No keypoints, cannot match")


if __name__ == "__main__":
    options = utilstest.options
    args = utilstest.args
    dirname = os.path.dirname(os.path.abspath(__file__))
    rep = os.path.join(dirname, "testimages")
    files = [ os.path.join(rep, f) for f in os.listdir(rep) if os.path.isfile(os.path.join(rep, f)) ]
    speedups = numpy.zeros((numpy.array(files).shape[0], 3), dtype=numpy.float32)
    i = 0

    for filename in files[:]:

        print("****** processing image %s ********" % (filename))
        d = DemoSift(fname=filename, context=utilstest.ctx)
        kp_ocl = d.sift_ocl()
        if feature:
            kp_cpp = d.sift_cpp()
        speedups[i] = d.timings(kp_ocl)
        i += 1
        if SHOW_FIGURES == True: d.show(1000)
#        d.match()
#        print "Done. Press any key..."

#        raw_input()
    print "Speedup \t Image size \t Number of keypoints"
    print speedups
#    print ""
#    print files

    sp_resolution = speedups[speedups[:, 1].argsort()]
    sp_keypoints = speedups[speedups[:, 2].argsort()]


    f1 = pylab.figure()
    sp1 = f1.add_subplot(111)
    sp1.plot(sp_resolution[:, 1] * 1e-6, sp_resolution[:, 0], "bs-")
    sp1.set_xlabel("Image resolution (Mega pixels)")
    sp1.set_ylabel("Speedup over CPU")
    #pylab.show()

    f2 = pylab.figure()
    sp2 = f2.add_subplot(111)
    sp2.plot(sp_keypoints[:, 2], sp_keypoints[:, 0], "rs-")
    sp2.set_xlabel("Number of keypoints found")
    sp2.set_ylabel("Speedup over CPU")
    pylab.show()

    raw_input()
