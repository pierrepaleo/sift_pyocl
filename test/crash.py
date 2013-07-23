#!/usr/bin/python
import time, sys, os
from math import sin, cos
here = os.path.dirname(os.path.abspath(__file__))
there = os.path.join(here, "..", "build")
lib = [os.path.abspath(os.path.join(there, i)) for i in os.listdir(there) if "lib" in i][0]
sys.path.insert(0, lib)
import sift
import numpy
import scipy.misc
import pylab
lena2 = scipy.misc.lena()
#lena = scipy.misc.imread("../stream.tiff") #for other tests
lena = numpy.ascontiguousarray(lena2[0:512,0:512])
# lena[:] = 0
# lena[100:110, 100:110] = 255
s = sift.SiftPlan(template=lena, profile=True, max_workgroup_size=128,devicetype="GPU",device=(1,0))
kpg = s.keypoints(lena)
kp = numpy.empty((kpg.size, 4), dtype=numpy.float32)
kp[:, 0] = kpg.x
kp[:, 1] = kpg.y
kp[:, 2] = kpg.scale
kp[:, 3] = kpg.angle



s.log_profile()
fig = pylab.figure()
sp1 = fig.add_subplot(1, 2, 1)
im = sp1.imshow(lena, cmap="gray")
sp1.set_title("OpenCL: %s keypoint" % kp.shape[0])
sp2 = fig.add_subplot(1, 2, 2)
im = sp2.imshow(lena, cmap="gray")


def cmp(a,b):
    if a.scale<b.scale:
        return True
    elif a.scale>b.scale:
        return False
    else:
        if a.angle>b.angle:
            return True
        else:
            return False


import feature
sc = feature.SiftAlignment()
res = sc.sift(lena)
ref = numpy.empty((res.size, 4), dtype=numpy.float32)
ref[:, 0] = res.x
ref[:, 1] = res.y
ref[:, 2] = res.scale
ref[:, 3] = res.angle

#numpy.savetxt("opencl_keypoints.txt",kp[numpy.argsort(kpg.y)],fmt='%.4f')
#numpy.savetxt("cpp_keypoints.txt",ref[numpy.argsort(res.y)],fmt='%.4f')
numpy.savetxt("opencl_descriptors_4.txt",(kpg[numpy.argsort(kpg.y)]).desc,fmt='%d')
numpy.savetxt("cpp_descriptors_sort_4.txt",(res[numpy.argsort(res.y)]).desc,fmt='%d')

#numpy.savetxt("opencl_cpu_cpukernels_not_sort.txt",(kpg).desc,fmt='%d')

#numpy.savetxt("cpp_descriptors_kp.txt",(ref[numpy.argsort(ref[:,1])]),fmt='%.3f')
#numpy.savetxt("opencl_cpukernels_kp.txt",(kp[numpy.argsort(kp[:,1])]),fmt='%.3f')



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

lres = list(res)
lres.sort(cmp)
lkpg = list(kpg)
lkpg.sort(cmp)
#print numpy.array([i.angle for i in lkpg[:250]])-numpy.array([i.angle for i in lres[:250]])

#res2 = numpy.recarray(shape=res.shape[0]*2,dtype=res.dtype)
#res2[:res.shape[0]] = res
#res2[res.shape[0]:] = res
t0_match = time.time()
match = feature.sift_match(res, kpg)
t1_match = time.time()
print "Angle1 - Angle0"
print abs(match.angle1-match.angle0).max()
#print match
print match.shape
print("Matching took %.3f ms" %(1000.0*(t1_match-t0_match)))
fig.show()


fig.show()
raw_input()

