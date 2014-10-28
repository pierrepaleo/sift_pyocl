#!/usr/bin/python
import sys, os
import utilstest
logger = utilstest.getLogger()
#here = os.path.dirname(os.path.abspath(__file__))
#there = os.path.join(here, "..", "build")
#lib = [os.path.abspath(os.path.join(there, i)) for i in os.listdir(there) if "lib" in i][0]
#sys.path.insert(0, lib)
import sift_pyocl as sift
import numpy
import scipy.misc

#
try:
    lena = scipy.misc.imread(sys.argv[1])
except:
    try:
        import fabio
        lena = fabio.open(sys.argv[1]).data
    except:
        lena = scipy.misc.lena()
        logger.info("Using image lena from scipy")
    else:
        logger.info("Using image %s read with FabIO"%sys.argv[1])
else:
    logger.info("Using image %s read with SciPy"%sys.argv[1])

s = sift.SiftPlan(template=lena, profile=True, context=utilstest.ctx)
kp = s.keypoints(lena)
print kp[:10]
if utilstest.args and not os.path.isfile(utilstest.args[0]):
    s.log_profile(utilstest.args[0])
else:
    s.log_profile()
