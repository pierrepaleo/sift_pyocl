Project structure
=================


Programming language
--------------------

SIFT_PyOCL uses the following programming languages:

* 6100 lines of Python (with the tests)
* 2800 lines of OpenCL (C-based language)

Repository:
-----------

The project is hosted by GitHub:
https://github.com/kif/sift_pyocl

Which provides the issue tracker in addition to Git hosting.
Collaboration is done via Pull-Requests in github's web interface.

Run dependencies
----------------

* python 2.6 or 2.7
* numPy
* sciPy
* matplotlib
* pyOpencl

Build dependencies:
-------------------
In addition to the run dependencies, SIFT_PyOCL needs an OpenCL compiler.


Building procedure
------------------

As most of the python projects:
...............................

    python setup.py build

Test suites
-----------

..

    python setup.py build test

Test suites ensure both non regression over time and ease the distribution under different platforms. As OpenCL compilers are different for each platform (AMD, Intel, Nvidia), it is crucial to check if the process is done correctly.


SIFT_PyOCL comes with test-suites for different stages of the process. All the OpenCL kernels (23 in total) are tested and compared to a Python implementation. For the last steps (keypoints orientation assignment and descriptor computation), the Python implementation is slow ; the ASIFT version can be used for the comparisons (see https://github.com/kif/imageAlignment/tree/numpy).

To run all the tests, use ``python test/test_all.py``. 


.. csv-table:: List of test suites
    :header: "Name", "Stmts", "Miss", "Cover"
    :widths: 50, 8, 8, 8
   
    "sift/__init__ ","8   ","0","100%"
    "sift/alignment   ","252   ","18  ","13%"
    "sift/match  ","187  ","152 ","19%"
    "sift/opencl ","184  ","77 ","58%"
    "sift/param","6","1 ","83%"
    "sift/plan   ","456  ","414 ","9%"
    "sift/sift ","0","0  ","0%"
    "sift/utils   ","69  ","51 ","26%"


Using the test suites
.....................


Folder structure
****************

Each test file in the ``test/`` folder aims at testing an OpenCL kernels files. For example, ``test/test_image.py`` is a test suite for ``openCL/image.cl``. All the OpenCL kernels have a reference implementation in Python, in file ``test/test_image_functions.py`` -- that is, SIFT is almost entirely re-written in Python, however, the functions are not written to be efficient.

For the last steps (orientation assignment, descriptors computation, matching), the Python implementation becomes slow because all the previous functions have to be called. Instead, one can use the ``feature`` module available here_.

.. _here: https://github.com/kif/imageAlignment/tree/numpy

It contains a Python wrapper for a SIFT C++ implementation. After installing it (``python setup.py build_ext --inplace``), you can get the keypoints with

.. code-block:: python

   import feature, scipy.misc
   l=scipy.misc.lena().astype("float32")
   s = feature.SiftAlignment()
   u = s.sift(l);

This module can be used to compare SIFT_PyOCL with ``feature.SiftAlignment`` results. SIFT_PyOCL was based on this implementation.


Customizing tests parameters
****************************

In the file ``test/test_image_setup.py``, SIFT parameters can be modified. The tests are run on a single scale of one octave, this can be modified as well.

.. csv-table:: SIFT parameters
    :header: "Parameter name" , "Default value" , "Description"
    :widths: 18, 18, 40

    "border_dist",   "5",              "Distance to the border. The pixels located at ``border_dist``" 
    "peakthresh",         "255.0*0.04/3.0",     "Threshold for the gray scale. Pixels whose grayscale is below will be ignored."
    "EdgeThresh",         "0.06",               "Threshold for the ratio of principal curvatures when testing if point lies on an edge"
    "EdgeThresh0",        "0.08",               "Threshold for the ratio of principal curvatures(first octave)"
    "doubleimsize",       "0",                  "The pre-blur factor is :math:`\sqrt{\sigma_0^2 - c^2}` with ``c = 0.5`` if ``doubleimsize = 0``, ``1.0`` otherwise "
    "initsigma",         "1.6",                "Initial blur factor (standard deviation of gaussian kernel)"
    "nb_keypoints",       "1000",               "Maximum number of keypoints, for buffers allocating"
    "ocsize",             "1",                  "Initially 1, then twiced at each octave. It is a power of two"
    "scale",              "1",                  "``scale`` can be 1, 2 or 3. Any other value is invalid !"


Additionally, the test image can be modified. Default is ``l2 = scipy.misc.lena().astype(numpy.float32)``. You can also specify the device to run on, at the bottom of ``test/utilstest.py`` :  ``ctx = ocl.create_context("GPU")``. Simply remplace "GPU" by "CPU" will run all the tests on the CPU.

The test suites files can have the following constant defined at the top of the file.

.. csv-table:: Default options, mangled in the  
    :header: "Constant name","Description"
    :widths: 18, 60

    "SHOW_FIGURES",       "If True, displays the figures with matplotlib                                 "
    "PRINT_KEYPOINTS",    "If True, displays parts of the keypoints vector for debugging                 "
    "USE_CPU",            "If True, runs the tests on CPU                                                "
    "USE_CPP_SIFT",       "If True, uses ``feature`` module for keypoints comparison instead of python   "


To fasten the tests, you can choose ``octsize = 4`` and ``scale = 1`` for example, as there are certainly less keypoints found in the superior octaves.






























