import sys, os
sys.path.insert(0,os.path.abspath("build/lib.linux-x86_64-2.7"))
import sift
import numpy
import scipy.misc
lena = scipy.misc.lena()
s = sift.SiftPlan(template=lena, profile=False, device=(0, 0))
out = s.keypoints(lena)
print "Still alive"
import Image
img = Image.open("/users/kieffer/Pictures/2010-01-21/17h51m32-Canon_PowerShot_G11.jpg")
nimg = numpy.fromstring(img.tostring(), "uint8")
nimg.shape = img.size[1], img.size[0], 3
ss = sift.SiftPlan(template=nimg)
print "Memory: %.3fMo" % (ss.memory / 2.0 ** 20)
ss.keypoints(nimg)


s = sift.SiftPlan(template=lena, profile=False, device=(0, 0))
s.keypoints(lena)
print "Still alive"
s.keypoints(lena)
print("Not yet dead")
import feature
#sc = feature.SiftAlignment()
#ref = sc.sift(lena)[:, :2]
#print ref[:10]
##s.keypoints(lena)
#for octave in range(s.octave_max):
#    data = s.buffers[(octave, "Kp")].get()
#    mask = data[:, 1] != -1
#    print data[mask, ][:10, 1:3]

