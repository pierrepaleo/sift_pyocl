#!/usr/bin/python
from utilstest import UtilsTest, getLogger
logger = getLogger(__file__)
import sift_pyocl as sift
import numpy
import scipy.misc
from math import  sin, cos
import pylab
from matplotlib.patches import ConnectionPatch

try:
    import feature
except:
    logger.error("Feature is not available to compare results with C++ implementation")
    feature = None
    res = numpy.empty(0)

img1 = scipy.misc.imread("testimages/img1.jpg")
img2 = scipy.misc.imread("testimages/img2.jpg")
plan = sift.SiftPlan(template=img1, devicetype="gpu")
kp1 = plan.keypoints(img1)
kp2 = plan.keypoints(img2)

fig = pylab.figure()
sp1 = fig.add_subplot(122)
sp2 = fig.add_subplot(121)
sp1.imshow(img1)
sp2.imshow(img2)
match = feature.sift_match(kp1, kp2)
fig.show()
#raw_input("Enter")
for m in match:
    con = ConnectionPatch(xyA=(m["x0"], m["y0"]), xyB=(m["x1"], m["y1"]), coordsA="data", coordsB="data", axesA=sp1, axesB=sp2, color="red")
    sp1.add_artist(con)
#    sp2.add_artist(con)

    x = m["x0"]
    y = m["y0"]
    scale = m["scale0"]
    angle = m["angle0"]
    x0 = x + scale * cos(angle)
    y0 = y + scale * sin(angle)
    sp1.annotate("", xy=(x, y), xytext=(x0, y0), color="blue",
                     arrowprops=dict(facecolor='blue', edgecolor='blue', width=1),)

    x = m["x1"]
    y = m["y1"]
    scale = m["scale1"]
    angle = m["angle1"]
    x0 = x + scale * cos(angle)
    y0 = y + scale * sin(angle)
    sp2.annotate("", xy=(x, y), xytext=(x0, y0), color="blue",
                     arrowprops=dict(facecolor='blue', edgecolor='blue', width=1),)

pylab.show()
s
