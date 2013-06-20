import sys, os
from math import sin, cos
here = os.path.dirname(os.path.abspath(__file__))
there = os.path.join(here,"..","build")
lib = [os.path.abspath(os.path.join(there,i)) for i in os.listdir(there) if "lib" in i][0]
sys.path.insert(0, lib)
import sift
import numpy
import scipy.misc
import pylab
lena = scipy.misc.lena()
s = sift.SiftPlan(template=lena, profile=True, max_workgroup_size=8)
kp = s.keypoints(lena)
s.log_profile()
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
res = sc.sift(lena)

ref = res[:, :4]

sp2.set_title("C++: %s keypoint" % ref.shape[0])
for i in range(ref.shape[0]):
    print res[i, 4:].max()
    x = ref[i, 0]
    y = ref[i, 1]
    scale = ref[i, 2]
    angle = ref[i, 3]
    sp2.annotate("", xy=(x, y), xytext=(x + scale * cos(angle), y + scale * sin(angle)), color="red",
                     arrowprops=dict(facecolor='red', edgecolor='red', width=1),)
    sp1.annotate("", xy=(x, y), xytext=(x + scale * cos(angle), y + scale * sin(angle)), color="blue",
                     arrowprops=dict(facecolor='blue', edgecolor='blue', width=1),)
fig.show()
print res[:, 4:].max()
#minkp = min(kp.shape[0], ref.shape[0])
#kpp = numpy.empty((minkp, 2, 2), dtype=numpy.float32)
#kpp[:, :, 0] = kp[:minkp, :2]
#kpp[:, :, 1] = ref[:minkp, :2]
#mateched = feature.reduce_orsa(kpp)
#print mateched.shape
#print mateched[:, 0:2] - mateched[:, 1:2]
#for y in mateched[:, 1]:
#    print y
#    r1 = numpy.where(abs(kp[:, 0] - y) < 0.001)[0][0]
#    r2 = numpy.where(abs(ref[:, 0] - y) < 0.001)[0][0]
#    print r1, r2
#
#    if abs(kp[r1] - ref[r2])[:3].max() > 0.1:
#        print kp[r1]
#        print ref[r2]
#        print kp[r1] - ref[r2]
for p0 in range(kp.shape[0]):
    best = sys.maxint
    best_id = -1
    kpi = kp[p0]
    for p1 in range(ref.shape[0]):
        refj = ref[p1]
        d = ((kpi - refj) ** 2).sum()
        if d < best:
            best = d,
            best_id = p1
    d = ((kpi - ref[best_id]) ** 2).sum()
    if d > 1:
        print kpi, (kpi - ref[best_id]).astype(int)

    sp1.annotate("", xy=kpi[:2], xytext=ref[best_id][:2], color="green",
                     arrowprops=dict(facecolor='green', edgecolor='green', width=1),)




fig.show()
raw_input()
