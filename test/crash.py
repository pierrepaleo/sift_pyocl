import sys, os
from math import sin, cos
sys.path.insert(0, os.path.abspath("build/lib.linux-x86_64-2.6"))
sys.path.insert(0, os.path.abspath("build/lib.linux-x86_64-2.7"))
import sift
import numpy
import scipy.misc
import pylab
lena = scipy.misc.lena()
s = sift.SiftPlan(template=lena, profile=False, devicetype="GPU", max_workgroup_size=8)
kp = s.keypoints(lena)
fig = pylab.figure()
sp1 = fig.add_subplot(1, 2, 1)
im = sp1.imshow(lena, cmap="gray")
sp1.set_title("OpenCL: %s keypoint" % kp.shape[0])
sp2 = fig.add_subplot(1, 2, 2)

im = sp2.imshow(lena, cmap="gray")

for i in range(kp.shape[0]):
    x = kp[i, 0]
    y = kp[i, 1]
    scale = kp[i, 2]
    angle = kp[i, 3]
    sp1.annotate("", xy=(x, y), xytext=(x + scale * cos(angle), y + scale * sin(angle)), color="red",
                     arrowprops=dict(facecolor='red', edgecolor='red', width=1),)


import feature
sc = feature.SiftAlignment()
ref = sc.sift(lena)[:, :4]
sp2.set_title("C++: %s keypoint" % ref.shape[0])
for i in range(ref.shape[0]):
    x = ref[i, 0]
    y = ref[i, 1]
    scale = ref[i, 2]
    angle = ref[i, 3]
    sp2.annotate("", xy=(x, y), xytext=(x + scale * cos(angle), y + scale * sin(angle)), color="red",
                     arrowprops=dict(facecolor='red', edgecolor='red', width=1),)
fig.show()
