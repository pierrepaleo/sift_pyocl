#!/usr/bin/python
import sys, os
here = os.path.dirname(os.path.abspath(__file__))
there = os.path.join(here, "..", "build")
lib = [os.path.abspath(os.path.join(there, i)) for i in os.listdir(there) if "lib" in i][0]
sys.path.insert(0, lib)
import sift
import numpy
import scipy.misc

#lena = scipy.misc.lena()
shape = 1001, 1599
lena2 = scipy.misc.imread("/users/kieffer/Pictures/2010-01-21/17h51m32-Canon_PowerShot_G11.jpg")#, flatten=True)
lena = numpy.ascontiguousarray(lena2[:shape[0], 0:shape[1], :])

s = sift.SiftPlan(template=lena, profile=True, devicetype="GPU")
kp = s.keypoints(lena)
