# -*- coding: utf-8 -*-
# /*##########################################################################
# Copyright (C) 2017 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ############################################################################*/
"""
Project: Sift implementation in Python + OpenCL
         https://github.com/silx-kit/silx
"""

from __future__ import division

__authors__ = ["Jérôme Kieffer", "Pierre Paleo"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "06/09/2017"
__status__ = "Production"

import os
import numpy
#from .. import resources
from math import log, ceil


def calc_size(shape, blocksize):
    """
    Calculate the optimal size for a kernel according to the workgroup size
    """
    if "__len__" in dir(blocksize):
        return tuple((int(i) + int(j) - 1) & ~(int(j) - 1) for i, j in zip(shape, blocksize))
    else:
        return tuple((int(i) + int(blocksize) - 1) & ~(int(blocksize) - 1) for i in shape)


def kernel_size(sigma, odd=False, cutoff=4):
    """
    Calculate the optimal kernel size for a convolution with sigma

    :param sigma: width of the gaussian
    :param odd: enforce the kernel to be odd (more precise ?)
    """
    size = int(ceil(2 * cutoff * sigma + 1))
    if odd and size % 2 == 0:
        size += 1
    return size




def nextpower(n):
    """Calculate the power of two

    :param n: an integer, for example 100
    :return: another integer, 100-> 128
    """
    return 1 << int(ceil(log(n, 2)))


def sizeof(shape, dtype="uint8"):
    """
    Calculate the number of bytes needed to allocate for a given structure

    :param shape: size or tuple of sizes
    :param dtype: data type
    """
    itemsize = numpy.dtype(dtype).itemsize
    cnt = 1
    if "__len__" in dir(shape):
        for dim in shape:
            cnt *= dim
    else:
        cnt = int(shape)
    return cnt * itemsize


def get_cl_file(resource):
    """get the full path of a openCL resource file

    The resource name can be prefixed by the name of a resource directory. For
    example "silx:foo.png" identify the resource "foo.png" from the resource
    directory "silx".
    See also :func:`silx.resources.register_resource_directory`.

    :param str resource: Resource name. File name contained if the `opencl`
        directory of the resources.
    :return: the full path of the openCL source file
    """
    if not resource.endswith(".cl"):
        resource += ".cl"
    return resources._resource_filename(resource,
                                        default_directory="opencl")


def read_cl_file(filename):
    """
    :param filename: read an OpenCL file and apply a preprocessor
    :return: preprocessed source code
    """
    with open(get_cl_file(filename), "r") as f:
        # Dummy preprocessor which removes the #include
        lines = [i for i in f.readlines() if not i.startswith("#include ")]
    return "".join(lines)




def bin2RGB(img):
    """
    Perform a 2x2 binning of the image
    """
    dtype = img.dtype
    if dtype == numpy.uint8:
        out_dtype = numpy.int32
    else:
        out_dtype = dtype
    shape = img.shape
    if len(shape) == 3:
        new_shape = shape[0] // 2, shape[1] // 2, shape[2]
        new_img = img
    else:
        new_shape = shape[0] // 2, shape[1] // 2, 1
        new_img = img.reshape((shape[0], shape[1], 1))
    out = numpy.zeros(new_shape, dtype=out_dtype)
    out += new_img[::2, ::2, :]
    out += new_img[1::2, ::2, :]
    out += new_img[1::2, 1::2, :]
    out += new_img[::2, 1::2, :]
    out /= 4
    if len(shape) != 3:
        out.shape = new_shape[0], new_shape[1]
    if dtype == numpy.uint8:
        return out.astype(dtype)
    else:
        return out


def matching_correction(matching):
    '''
    Given the matching between two list of keypoints, 
    return the linear transformation to correct kp2 with respect to kp1
    '''
    N = matching.shape[0]
    #solving normals equations for least square fit
    #
    # We correct for linear transformations, mapping points (x, y)
    # to points (x', y') :
    #
    #   x' = a*x + b*y + c
    #   y' = d*x + e*y + f
    #
    # where the parameters a, ..., f  determine the linear transformation.
    # The equivalent matrix form is
    #
    #   x1  y1  1   0   0   0           a       x1'
    #   0   0   0   x1  y1  1           b       y1'
    #   x2  y2  1   0   0   0     x     c   =   x2'
    #   0   0   0   x2  y2  1           d       y2'
    #       . . . . . .                 e       .
    #                                   f       .
    X = numpy.zeros((2 * N, 6))
    X[::2, 2:] = 1, 0, 0, 0
    X[::2, 0] = matching.x[:, 0]
    X[::2, 1] = matching.y[:, 0]
    X[1::2, 0:3] = 0, 0, 0
    X[1::2, 3] = matching.x[:, 0]
    X[1::2, 4] = matching.y[:, 0]
    X[1::2, 5] = 1
    y = numpy.zeros((2 * N, 1))
    y[::2, 0] = matching.x[:, 1]
    y[1::2, 0] = matching.y[:, 1]

get_opencl_code = read_cl_file


def concatenate_cl_kernel(filenames):
    """Concatenates all the kernel from the list of files

    :param filenames: filenames containing the kernels
    :type filenames: list of str which can be filename of kernel as a string.
    :return: a string with all kernels concatenated

    this method concatenates all the kernel from the list
    """
    return os.linesep.join(read_cl_file(fn) for fn in filenames)
