#!/usr/bin/python
from utilstest import UtilsTest, getLogger
logger = getLogger(__file__)
import sys
import sift_pyocl as sift
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
    try:
        img1 = scipy.misc.imread(sys.argv[1])
        img2 = scipy.misc.imread(sys.argv[2])
    except IOError:
        try:
            import fabio
            img1 = fabio.open(sys.argv[1]).data
            img2 = fabio.open(sys.argv[2]).data
        except:
            logger.error("Unable to read input images")
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
plan = sift.SiftPlan(template=img1, devicetype="cpu")
kp1 = plan.keypoints(img1)
kp2 = plan.keypoints(img2)
print("Keypoints for img1: %i\t img2: %i" % (kp1.size, kp2.size))
fig = pylab.figure()
sp1 = fig.add_subplot(122)
sp2 = fig.add_subplot(121)
im1 = sp1.imshow(img1)
im2 = sp2.imshow(img2)
#match = feature.sift_match(kp1, kp2)
mp = sift.MatchPlan(devicetype="cpu")
matching = mp.match(kp1, kp2)
print("After matching keeping %i" % matching.shape[0])
dx = matching[:, 1].x - matching[:, 0].x
dy = matching[:, 1].y - matching[:, 0].y
dangle = matching[:, 1].angle - matching[:, 0].angle
dscale = numpy.log(matching[:, 1].scale / matching[:, 0].scale)
distance = numpy.sqrt(dx * dx + dy * dy)
outlayer = numpy.zeros(distance.shape, numpy.int8)
cutoff = 1.1
cont = True
outlayersum = matching.shape[0]
while matching.shape[0]-outlayersum <= 6:
    outlayer += abs((distance - distance.mean()) / distance.std()) > cutoff
    outlayer += abs((dangle - dangle.mean()) / dangle.std()) > cutoff
    outlayer += abs((dscale - dscale.mean()) / dscale.std()) > cutoff
    outlayersum = (outlayer > 0).sum()
    print("Found %i outlayers with cutoff at %.1f std"%( outlayersum,cutoff))
    cutoff+=1.0
if outlayersum > 0 and not numpy.isinf(outlayersum):
    matching = matching[outlayer == 0]

fig.show()
#raw_input("Enter")
for i in range(matching.shape[0]):
    k1 = matching[i, 0]
    k2 = matching[i, 1]
    con = ConnectionPatch(xyA=(k1.x, k1.y), xyB=(k2.x, k2.y), coordsA="data", coordsB="data", axesA=sp1, axesB=sp2, color="blue")
    con2 = ConnectionPatch(xyA=(k2.x, k2.y), xyB=(k1.x, k1.y), coordsA="data", coordsB="data", axesA=sp2, axesB=sp1, color="blue")
    sp1.add_artist(con)
    sp2.add_artist(con2)

pylab.show()

