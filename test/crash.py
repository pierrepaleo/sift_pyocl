import sys, os
sys.path.insert(0,os.path.abspath("build/lib.linux-x86_64-2.6"))
import sift
import scipy.misc
lena = scipy.misc.lena()
s = sift.SiftPlan(template=lena,profile=False,device=(0,0))
s.keypoints(lena)
print "Still alive"
s.keypoints(lena)
print("Not yet dead")
import feature
sc = feature.SiftAlignment()
ref = sc.sift(lena)[:, :2]
print ref[:10]
#s.keypoints(lena)
for octave in range(s.octave_max):
    data = s.buffers[(octave, "Kp")].get()
    mask = data[:, 1] != -1
    print data[mask, ][:10, 1:3]

