General introduction to SIFT_PyOCL
==================================

SIFT_PyOCL, a parallel version of SIFT algorithm
------------------------------------------------

SIFT (Scale-Invariant Feature Transform) is an algorithm developped by David Lowe in 1999. It is a worldwide reference for image alignment and object recognition. The robustness of this method enables to detect features at different scales, angles and illumination of a scene. SIFT_PyOCL is an implementation of SIFT in OpenCL, meaning that it can run on Graphics Processing Units and Central Processing Units as well. Interest points are detected in the image, then data structures called *descriptors* are built to be characteristic of the scene, so that two different images of the same scene have similar descriptors. They are robust to transformations like translation, rotation, rescaling and illumination change, which make SIFT interesting for image stitching. In the fist stage, descriptors are computed from the input images. Then, they are compared to determine the geometric transformation to apply in order to align the images. SIFT_PyOCL can run on most graphic cards and CPU, making it usable on many setups. OpenCL processes are handled from Python with PyOpenCL, a module to access OpenCL parallel computation API.



Introduction
------------

The European Synchrotron Radiation Facility (ESRF) beamline ID21 developed a full-field method for X-ray absorption near-edge spectroscopy (XANES). Since the flat field images are not acquired simultaneously with the sample transmission images, a realignment procedure has to be performed. SIFT is currently used, but takes about 8 seconds per frame, and one stack can have up to 500 frames. It is a bottleneck in the global process, therefore a parallel version of this algorithm can provide a crucial speed-up.




SIFT descriptors computation
----------------------------

Before image alignment, descriptors have to be computed from each image. The whole process can be launched by several lines of code.


How to use it
.............

SIFT_PyOCL can be installed as a standard Debian package, or with "python setup.py build". It generates a library that can be imported, then used to compute a list of descriptors from an image. The image can be in RGB values, but all the process is done on grayscale values. One can specify the device, either CPU or GPU. Although being integrated in EDNA framework for online image alignment, and thus mostly used by developers, SIFT_PyOCL provides example scripts.

.. code-block:: python

   python test/demo.py --type=GPU my_image.jpg

This computes and shows the keypoints on the input image.
One can also launch SIFT_PyOCL interactively with iPython :

.. _example1:
.. code-block:: python

   import sift
   import numpy
   import scipy.misc
   image_rgb = scipy.misc.imread("../my_image.jpg")
   sift_ocl = sift.SiftPlan(template=image_rgb, device=GPU)
   kp = sift_ocl.keypoints(image_rgb)
   kp.sort(order=["scale", "angle", "x", "y"])
   print kp




SIFT_PyOCL Files
................

The Python sources are in the "sift-src" folder. The file "plan.py" executes the whole process, from kernel compilation to descriptors computation. The OpenCL kernels in the "openCL" folder are compiled on the fly. Several kernels have multiple implementations, depending the architecture to run on.



~~~~~~~~~~~~~

The different steps of SIFT are handled by ``plan.py``.






Image matching and alignment
----------------------------



There is a demo file "demo_match.py" that can be run to have a keypoints matching demonstration. Matching can also be run from ipython : suppose we got two list of keypoints "kp1" and "kp2" according to example1_.

.. _example2:
.. code-block:: python

   mp = sift.MatchPlan()
   match = mp.match(kp1, kp2)
   print("Number of Keypoints with for image 1 : %i, For image 2 : %i, Matching keypoints: %i" % (kp1.size, kp2.size, match.shape[0]))



Performances
------------

The aim of SIFT_PyOCL is to fasten the image alignment by running it on GPU.


.. figure:: img/bench_gpu_res.png
   :align: center
   :alt: Benchmark GPU vs CPU




 















References
..........

- David G. Lowe, Distinctive image features from scale-invariant keypoints, International Journal of Computer Vision, vol. 60, no 2, 2004, p. 91–110
http://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf


