#!/usr/bin/python
import sys, os, pyopencl
from math import sin, cos
here = os.path.dirname(os.path.abspath(__file__))
there = os.path.join(here, "..", "build")
lib = [os.path.abspath(os.path.join(there, i)) for i in os.listdir(there) if "lib" in i][0]
sys.path.insert(0, lib)
import sift_pyocl as sift
import numpy
import scipy.misc
import pylab
lena2 = scipy.misc.lena()
#shape = lena2.shape
#lena2 = scipy.misc.imread("../aerial.tiff") #for other tests
#filename = "/users/kieffer/Pictures/2010-01-22/11h59m14-Canon_DIGITAL_IXUS_850_IS-Pano21.jpg"
#shape = 1001, 1599
#lena2 = scipy.misc.imread(filename)#, flatten=True)
#shape = lena2.shape
#lena = numpy.ascontiguousarray(lena2[:shape[0], 0:shape[1], :])
lena = lena2
print lena.shape

# lena[:] = 0
# lena[100:110, 100:110] = 255
s = sift.SiftPlan(template=lena, profile=True, max_workgroup_size=128, device=(1, 0))
kpg = s.keypoints(lena)
#except (Exception, pyopencl.RuntimeError) as error:
#    print error
#    s.log_profile()
#    sys.exit(0)
kp = numpy.empty((kpg.size, 4), dtype=numpy.float32)
kp[:, 0] = kpg.x
kp[:, 1] = kpg.y
kp[:, 2] = kpg.scale
kp[:, 3] = kpg.angle
#print "Non infinite", numpy.isfinite(kpg.angle).sum()
s.log_profile()
fig = pylab.figure()
sp1 = fig.add_subplot(1, 2, 1)
im = sp1.imshow(lena, cmap="gray")
sp1.set_title("OpenCL: %s keypoint" % kp.shape[0])
sp2 = fig.add_subplot(1, 2, 2)
im = sp2.imshow(lena, cmap="gray")


def cmp(a, b):
    if a.scale < b.scale:
        return True
    elif a.scale > b.scale:
        return False
    else:
        if a.angle > b.angle:
            return True
        else:
            return False




import feature
sc = feature.SiftAlignment()
#lena2 = scipy.misc.imread(filename, flatten=True)
#lena = numpy.ascontiguousarray(lena2[:shape[0], :shape[1]])

res = sc.sift(lena)
ref = numpy.empty((res.size, 4), dtype=numpy.float32)
ref[:, 0] = res.x
ref[:, 1] = res.y
ref[:, 2] = res.scale
ref[:, 3] = res.angle

sp2.set_title("C++: %s keypoint" % ref.shape[0])
for i in range(ref.shape[0]):
#    print res[i, 4:].max()
    x = ref[i, 0]
    y = ref[i, 1]
    scale = ref[i, 2]
    angle = ref[i, 3]
    sp2.annotate("", xy=(x, y), xytext=(x + scale * cos(angle), y + scale * sin(angle)), color="red",
                     arrowprops=dict(facecolor='red', edgecolor='red', width=1),)
    sp1.annotate("", xy=(x, y), xytext=(x + scale * cos(angle), y + scale * sin(angle)), color="red",
                     arrowprops=dict(facecolor='red', edgecolor='red', width=1),)

for i in range(kp.shape[0]):
    x = kp[i, 0]
    y = kp[i, 1]
    scale = kp[i, 2]
    angle = kp[i, 3]
    sp1.annotate("", xy=(x, y), xytext=(x + scale * cos(angle), y + scale * sin(angle)), color="blue",
                     arrowprops=dict(facecolor='blue', edgecolor='blue', width=1),)
#print numpy.degrees((ref[numpy.argsort(res.scale)][:392] - kp[numpy.argsort(kpg.scale)][:392])[:,3])

#print res[:5]
#print ""*80
#print kpg[:5]
match = feature.sift_match(res, kpg)
#print match
print match.shape
fig.show()


fig.show()
raw_input()


