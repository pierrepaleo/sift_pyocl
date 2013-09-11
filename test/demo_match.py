#!/usr/bin/python
from utilstest import UtilsTest, getLogger
logger = getLogger(__file__)
import sys
import sift
import numpy
import scipy.misc, scipy.ndimage
from math import  sin, cos
import pylab
from matplotlib.patches import ConnectionPatch

try:
    import feature
except:
    logger.error("Feature is not available to compare results with C++ implementation")
    feature = None
    res = numpy.empty(0)
if len(sys.argv) == 3:
    img1 = scipy.misc.imread(sys.argv[1])
    img2 = scipy.misc.imread(sys.argv[2])
else:
    img1 = scipy.misc.lena()
    img2 = scipy.ndimage.rotate(img1, 10, reshape=False)
'''
img1 = scipy.misc.imread("../test_images/fruit_bowl.png")
tmp = scipy.misc.imread("../test_images/banana.png")
img2 = numpy.zeros_like(img1)
img2 = img2 + 255
d0 = (img1.shape[0] - tmp.shape[0])/2
d1 = (img1.shape[1] - tmp.shape[1])/2
img2[d0:-d0,d1:-d1-1] = numpy.copy(tmp)
'''       
plan = sift.SiftPlan(template=img1, devicetype="gpu")
kp1 = plan.keypoints(img1)
kp2 = plan.keypoints(img2)
print("Keypoints for img1: %i\t img2: %i" % (kp1.size, kp2.size))
fig = pylab.figure()
sp1 = fig.add_subplot(122)
sp2 = fig.add_subplot(121)
im1 = sp1.imshow(img1)
im2 = sp2.imshow(img2)
#match = feature.sift_match(kp1, kp2)
mp = sift.MatchPlan()
m = mp.match(kp1, kp2)
print("After matching keeping %i" % m.shape[0])
fig.show()
#raw_input("Enter")
for i in range(m.shape[0]):
    k1 = m[i, 0]
    k2 = m[i, 1]
    con = ConnectionPatch(xyA=(k1.x, k1.y), xyB=(k2.x, k2.y), coordsA="data", coordsB="data", axesA=sp1, axesB=sp2, color="blue")
    con2 = ConnectionPatch(xyA=(k2.x, k2.y), xyB=(k1.x, k1.y), coordsA="data", coordsB="data", axesA=sp2, axesB=sp1, color="blue")
    sp1.add_artist(con)
    sp2.add_artist(con2)

pylab.show()

