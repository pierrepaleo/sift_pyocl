# -*- coding: utf8 -*-
#
#    Project: Image Alignment
#
#
#    File: "$Id$"
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

__author__ = "Jerome Kieffer"
__license__ = "GPLv3"
__date__ = "27/05/2013"
__copyright__ = "2011-2012, ESRF"
__contact__ = "jerome.kieffer@esrf.fr"
__doc__ = "this is a cython wrapper for feature extraction algorithm"

import cython, time, threading, multiprocessing
from libc.string cimport memcpy
from libc.stdlib cimport free 
from cython.operator cimport dereference as deref
from cython.parallel cimport prange
from cpython.object cimport PyObject
import numpy
cimport numpy
from libcpp cimport bool
from libcpp.pair  cimport pair
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.list cimport list
from libc.stdint cimport uint64_t, uint32_t
from threading import Semaphore
from surf cimport  image, keyPoint, descriptor, listDescriptor, getKeyPoints, listKeyPoints, listMatch, octave, interval, matchDescriptor, get_points
from sift cimport  keypoint, keypointslist, default_sift_parameters, compute_sift_keypoints, siftPar, matchingslist, compute_sift_matches, compute_sift_keypoints_flimage, flimage, imgblur, sample
from asift cimport compute_asift_matches, compute_asift_keypoints
from orsa cimport Match, MatchList, orsa
from crc32 cimport crc32
from multiprocessing import cpu_count

dtype_kp = numpy.dtype([('x', numpy.float32), 
                        ('y', numpy.float32), 
                        ('scale', numpy.float32), 
                        ('angle', numpy.float32), 
                        ('desc', (numpy.uint8, 128))
                        ])

#cdef packed struct dtype_kp_t:
cdef packed struct dtype_kp_t:
    numpy.float32_t   x, y, scale, angle
    unsigned char     desc[128]

dtype_match = numpy.dtype([('x0', numpy.float32),('y0', numpy.float32),
                           ('scale0', numpy.float32), ('angle0', numpy.float32),
                           ('x1', numpy.float32),('y1', numpy.float32),
                           ('scale1', numpy.float32), ('angle1', numpy.float32)])
cdef packed struct dtype_match_t:
    numpy.float32_t   x0, y0, scale0, angle0, x1, y1, scale1, angle1

def mycrc(float[:] data):
    return crc32(< char *> & data[0], data.size * sizeof(float))


def normalize_image(numpy.ndarray img not None):
    maxi = numpy.float32(img.max())
    mini = numpy.float32(img.min())
    return numpy.ascontiguousarray(numpy.float32(255) * (img - mini) / (maxi - mini), dtype=numpy.float32)

cdef keypoints2array(keypointslist kpl):
    """
    Function that converts a keypoint list (n keypoints) into a numpy array of shape = (n, 132) 
    Each keypoint is composed of x, y, scale and angle and a vector containing 4*4*8 = 128 floats   
    """
    cdef int i,j, n = kpl.size()
#    cdef dtype_kp_t[:] out 
#    cdef numpy.ndarray[dtype_kp_t, ndim=1] out
    out = numpy.recarray(shape=(n,), dtype=dtype_kp) 
    cdef float[:] out_x  = numpy.zeros(n, dtype=numpy.float32)
    cdef float[:] out_y  = numpy.zeros(n, dtype=numpy.float32)
    cdef float[:] out_scale = numpy.zeros(n, dtype=numpy.float32)
    cdef float[:] out_angle  = numpy.zeros(n, dtype=numpy.float32)
    cdef unsigned char[:,:] out_desc  = numpy.zeros((n,128), dtype=numpy.uint8)
    #numpy.recarray(shape=(n,), dtype=dtype_kp)
#    cdef dtype_kp_t nkp 
    cdef keypoint kp 
    for i in range(n):
        kp = kpl[i]
        out_x[i] = kp.x
        out_y[i] = kp.y
        out_scale[i] = kp.scale
        out_angle[i] = kp.angle
        for j in range(128):
            out_desc[i,j] = <unsigned char> (kp.vec[j])
    out[:].x = out_x
    out[:].y = out_y
    out[:].scale = out_scale
    out[:].angle = out_angle
    out[:].desc = out_desc
    return out

cdef keypointslist array2keypoints(numpy.ndarray ary):
    """
    Function that converts a numpy array into keypoint list.
    The numpy array must have a second dimension equal to 132 !!!  
    Each keypoint is composed of x, y, scale and angle and a vector containing 4*4*8 = 128 floats   
    """
    assert ary.ndim == 1
    assert ary.dtype == dtype_kp
    cdef int i, j, n = ary.size
    cdef keypoint kp
    cdef dtype_kp_t nkp
    cdef keypointslist kpl # = new keypointslist()
    cdef float[:] x = ary[:].x
    cdef float[:] y = ary[:].y
    cdef float[:] scale = ary[:].scale
    cdef float[:] angle = ary[:].angle
    cdef unsigned char[:,:] desc = ary[:].desc 
    for i in range(n):
        kp.x = x[i] 
        kp.y = y[i] 
        kp.scale =  scale[i]
        kp.angle = angle[i]  
        for j in range(128):
            kp.vec[j] = desc[i,j]
        kpl.push_back(kp)  
    return kpl

def sift_keypoints(numpy.ndarray img): 
    """
    Calculate all keypoints from an image 
    
    @param image: 2D numpy array
    @return: 1D numpyarray of keypoints
    """   
    assert img.ndim == 2
    cdef float[:, :] data = normalize_image(img)
    cdef keypointslist kp
    cdef siftPar sift_parameters
    with nogil:
        default_sift_parameters(sift_parameters)
        compute_sift_keypoints(< float *> & data[0, 0], kp, data.shape[1], data.shape[0], sift_parameters)
    out = keypoints2array(kp)
    kp.empty()
#    del kp
#    del sift_parameters
    return out

def sift_match(numpy.ndarray nkp1,numpy.ndarray nkp2): 
    """
    Calculate all keypoints from an image 
    
    @param image: 2D numpy array
    @return: 1D numpyarray of keypoints
    """   
    assert nkp1.ndim == 1
    assert nkp1.ndim == 1
    assert type(nkp1) == numpy.core.records.recarray
    assert type(nkp2) == numpy.core.records.recarray
    cdef keypointslist kp1,kp2
    cdef matchingslist matchings
    cdef siftPar sift_parameters

    kp1 = array2keypoints(nkp1)
    kp2 = array2keypoints(nkp2)
    with nogil:
        default_sift_parameters(sift_parameters)
        compute_sift_matches(kp1, kp2, matchings, sift_parameters);
    cdef numpy.ndarray[dtype_match_t, ndim = 1] out = numpy.recarray(shape=(matchings.size(), ), dtype=dtype_match)
    for i in range(matchings.size()):
        out[i].x0 = matchings[i].first.x
        out[i].y0 = matchings[i].first.y
        out[i].scale0 = matchings[i].first.scale
        out[i].angle0 = matchings[i].first.angle
        out[i].x1 = matchings[i].second.x
        out[i].y1 = matchings[i].second.y
        out[i].scale1 = matchings[i].second.scale
        out[i].angle1 = matchings[i].second.angle
                
    matchings.empty()
    return out
    
    
cdef class SiftAlignment:
    cdef siftPar sift_parameters
    cdef map[uint32_t, keypointslist] dictKeyPointsList
    cdef FastRLock lock
    cdef object sem
#    cdef list[uint32_t] processing

    def __cinit__(self):
        default_sift_parameters(self.sift_parameters)
        self.dictKeyPointsList = map[uint32_t, keypointslist]()
        self.lock = FastRLock()
        self.sem = Semaphore(<int>cpu_count())
#        self.processing = list()
    
    def __dealloc__(self):
        self.clear()
        
    def clear(self):
        """
        Empty the vector of keypoints.
        """
        with self.lock:
            self.dictKeyPointsList.empty()
#            self.processing.empty()

    @cython.boundscheck(False)
    cdef keypointslist sift_c(self, numpy.ndarray img):
        """
        Calculate the SIFT descriptor for an image and stores it.
        Cython only version

        @param img: 2D numpy array representing the image
        @return: index of keypoints in the list
        """
        cdef float[:, :] data = normalize_image(img)
        cdef keypointslist kp
        cdef uint32_t idx = crc32(< char *> & data[0, 0], img.size * sizeof(float))
        if (self.dictKeyPointsList.find(idx)!=self.dictKeyPointsList.end()):
            return self.dictKeyPointsList[idx]
        with self.sem:
            t0 = time.time()
            with nogil:
                compute_sift_keypoints(< float *> & data[0, 0], kp, data.shape[1], data.shape[0], self.sift_parameters)
            print("SIFT on image %4ix%4i took %.3fms"%(img.shape[1],img.shape[0],1000.0*(time.time()-t0)))
        with self.lock:
            self.dictKeyPointsList[idx] = kp
        return kp

    @cython.boundscheck(False)
    def sift(self, numpy.ndarray img not None):
        """
        Calculate the SIFT descriptor for an image. Python version

        @param img: 2D numpy array representing the image
        @return: list keypoints as a numpy array
        """
        return keypoints2array(self.sift_c(img))


    @cython.boundscheck(False)
    def match(self, data1, data2):
        """
        calculate the matching between two images already analyzed.

        @param idx1, idx2: indexes of the images in the stored
        @return:   n x 4 numpy ndarray with [y1,x1,y2,x2] control points.
        """
        cdef keypointslist kp1 , kp2
        cdef matchingslist matchings
        if type(data1) == numpy.core.records.recarray and data1.ndim == 1:
            kp1 =  array2keypoints(data1)
            print kp1.size()
        if type(data2) == numpy.core.records.recarray and data2.ndim == 1:
            kp2 =  array2keypoints(data2)
            print kp2.size()
        with nogil:
            compute_sift_matches(kp1, kp2, matchings, self.sift_parameters);
        cdef numpy.ndarray[numpy.float32_t, ndim = 2] out = numpy.zeros((matchings.size(), 4), dtype=numpy.float32)
        for i in range(matchings.size()):
            out[i, 0] = matchings[i].first.y
            out[i, 1] = matchings[i].first.x
            out[i, 2] = matchings[i].second.y
            out[i, 3] = matchings[i].second.x
        return out

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def asift2(numpy.ndarray in1 not None, numpy.ndarray in2 not None, bool verbose=False):
    """
    Call ASIFT on a pair of images
    @param in1: first image
    @type in1: numpy ndarray
    @param in2: second image
    @type in2: numpy ndarray
    @param verbose: indicate the default verbosity
    @return: 2D array with n control points and 4 coordinates: in1_0,in1_1,in2_0,in2_1
    """
    cdef int i
    cdef int num_of_tilts1 = 7
    cdef int num_of_tilts2 = 7
    cdef int verb = < int > verbose
    cdef siftPar siftparameters
    default_sift_parameters(siftparameters)
    cdef vector[ vector[ keypointslist ]] keys1
    cdef vector[ vector[ keypointslist ]] keys2
    cdef int num_keys1 = 0, num_keys2 = 0
    cdef int num_matchings
    cdef matchingslist matchings

#    cdef vector [ float ] ipixels1_zoom, ipixels2_zoom
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] data1 = numpy.ascontiguousarray(255. * (in1.astype("float32") - in1.min()) / (in1.max() - in1.min()))
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] data2 = numpy.ascontiguousarray(255. * (in2.astype("float32") - in2.min()) / (in2.max() - in2.min()))
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] fdata1 = data1.flatten()
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] fdata2 = data2.flatten()
    cdef vector [ float ] ipixels1_zoom = vector [ float ](< size_t > data1.size)
    cdef vector [ float ] ipixels2_zoom = vector [ float ](< size_t > data2.size)
    for i in range(data1.size):
        ipixels1_zoom[i] = < float > fdata1[i]
    for i in range(data2.size):
        ipixels2_zoom[i] = < float > fdata2[i]

    if verbose:
        import time
        print("Computing keypoints on the two images...")
        tstart = time.time()
        num_keys1 = compute_asift_keypoints(ipixels1_zoom, data1.shape[1] , data1.shape[0] , num_of_tilts1, verb, keys1, siftparameters)
        tint = time.time()
        print "ASIFT took %.3fs image1: %i ctrl points" % (tint - tstart, num_keys1)
        num_keys2 = compute_asift_keypoints(ipixels2_zoom, data2.shape[1], data2.shape[0], num_of_tilts2, verb, keys2, siftparameters)
        tend = time.time()
        print "ASIFT took %.3fs image2: %i ctrl points" % (tend - tint, num_keys2)
        tend = time.time()
        num_matchings = compute_asift_matches(num_of_tilts1, num_of_tilts2,
                                              data1.shape[1] , data1.shape[0],
                                               data2.shape[1], data2.shape[0],
                                               verb, keys1, keys2, matchings, siftparameters)
        tmatch = time.time()
        print("Matching: %s point, took %.3fs " % (num_matchings, tmatch - tend))
    else:
        num_keys1 = compute_asift_keypoints(ipixels1_zoom, data1.shape[1] , data1.shape[0] , num_of_tilts1, verb, keys1, siftparameters)
        num_keys2 = compute_asift_keypoints(ipixels2_zoom, data2.shape[1], data2.shape[0], num_of_tilts2, verb, keys2, siftparameters)
        num_matchings = compute_asift_matches(num_of_tilts1, num_of_tilts2,
                                              data1.shape[1] , data1.shape[0],
                                               data2.shape[1], data2.shape[0],
                                               verb, keys1, keys2, matchings, siftparameters)

    cdef numpy.ndarray[numpy.float32_t, ndim = 2] out = numpy.zeros((num_matchings, 4), dtype="float32")
    matchings.begin()
    for i in range(matchings.size()):
        out[i, 0] = matchings[i].first.y
        out[i, 1] = matchings[i].first.x
        out[i, 2] = matchings[i].second.y
        out[i, 3] = matchings[i].second.x
    return out
    
    
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def reduce_orsa(numpy.ndarray inp not None, shape=None, bool verbose=False):
    """
    Call ORSA (keypoint checking)
    @param inp: n*4 ot n*2*2 array representing keypoints.
    @type in1: numpy ndarray
    @param shape: shape of the input images (unless guessed)
    @type shape: 2-tuple of integers
    @return: 2D array with n control points and 4 coordinates: in1_0,in1_1,in2_0,in2_1
    """

    cdef int i, num_matchings, insize, p
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] data = numpy.ascontiguousarray(inp.astype("float32").reshape(-1, 4))
    insize = data.shape[0]
    if insize < 10:
        return data
    cdef vector [ Match ]  match_coor = vector [ Match ](< size_t > insize)
    cdef int t_value_orsa = 10000
    cdef int verb_value_orsa = verbose
    cdef int n_flag_value_orsa = 0
    cdef int mode_value_orsa = 2
    cdef int stop_value_orsa = 0
    cdef float nfa
    cdef int width, heigh
    if shape is None:
        width = int(1 + max(data[:, 1].max(), data[:, 3].max()))
        heigh = int(1 + max(data[:, 0].max(), data[:, 2].max()))
    elif hasattr(shape, "__len__") and len(shape) >= 2:
        width = int(shape[1])
        heigh = int(shape[0])
    else:
        width = heigh = int(shape)
    cdef vector [ float ] index = vector [ float ](< size_t > data.shape[0])
    tmatch = time.time()
    with nogil:
        for i in range(data.shape[0]):
            match_coor[i].y1 = < float > data[i, 0]
            match_coor[i].x1 = < float > data[i, 1]
            match_coor[i].y2 = < float > data[i, 2]
            match_coor[i].x2 = < float > data[i, 3]
    # epipolar filtering with the Moisan - Stival ORSA algorithm.
        nfa = orsa(width, heigh, match_coor, index, t_value_orsa, verb_value_orsa, n_flag_value_orsa, mode_value_orsa, stop_value_orsa)
    tend = time.time()
    num_matchings = index.size()
    if verbose:
        print("Matching with ORSA: %s => %s, took %.3fs, nfs=%s" % (insize, num_matchings, tend - tmatch, nfa))
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] out = numpy.zeros((num_matchings, 4), dtype="float32")
    for i in range(index.size()):
        p = < int > index[i]
        out[i, 0] = data[p, 0]
        out[i, 1] = data[p, 1]
        out[i, 2] = data[p, 2]
        out[i, 3] = data[p, 3]
    return out

cdef void printCtrlPointSift(keypointslist kpt, int maxLines=10):
    """
    Print the control points
    """
    cdef int i
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] out = numpy.zeros((kpt.size(), 4), dtype="float32")
    for i in range(kpt.size()):
        out[i, 0] = kpt[i].x
        out[i, 1] = kpt[i].y
        out[i, 2] = kpt[i].scale
        out[i, 3] = kpt[i].angle
    out.sort(axis=0)
    for i in range(min(maxLines, kpt.size())):
        print out[i]

cdef void printMatching(matchingslist match, int maxLines=10):
    """
    Print the matching control points
    """
    cdef int i
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] out = numpy.zeros((match.size(), 4), dtype="float32")
    for i in range(match.size()):
        out[i, 0] = match[i].first.x
        out[i, 1] = match[i].first.y
        out[i, 2] = match[i].second.x
        out[i, 3] = match[i].second.y
    out.sort(axis=0)
    for i in range(min(maxLines, match.size())):
        print out[i]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def surf2(numpy.ndarray in1 not None, numpy.ndarray in2 not None, bool verbose=False):
    """
    Call surf on a pair of images
    @param in1: first image
    @type in1: numpy ndarray
    @param in2: second image
    @type in2: numpy ndarray
    @return: 2D array with n control points and 4 coordinates: in1_0,in1_1,in2_0,in2_1
    """
    cdef listKeyPoints * l1 = new listKeyPoints()
    cdef listKeyPoints * l2 = new listKeyPoints()
    cdef listDescriptor * listeDesc1
    cdef listDescriptor * listeDesc2
    cdef listMatch * matching

    cdef numpy.ndarray[numpy.float32_t, ndim = 2] data1 = numpy.ascontiguousarray(255. * (in1.astype("float32") - in1.min()) / (in1.max() - in1.min()))
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] data2 = numpy.ascontiguousarray(255. * (in2.astype("float32") - in2.min()) / (in2.max() - in2.min()))
    cdef image * img1 = new image(data1.shape[1], data1.shape[0])
    img1.img = < float *> data1.data
    cdef image * img2 = new image(data2.shape[1], data2.shape[0])
    img2.img = < float *> data2.data

    if verbose:
        import time
        time_init = time.time()
        listeDesc1 = getKeyPoints(img1, octave, interval, l1, verbose)
        time_int = time.time()
        print "SURF took %.3fs image1: %i ctrl points" % (time_int - time_init, listeDesc1.size())
        time_int = time.time()
        listeDesc2 = getKeyPoints(img2, octave, interval, l2, verbose)
        time_finish = time.time()
        print "SURF took %.3fs image2: %i ctrl points" % (time_finish - time_int, listeDesc2.size())
        time_finish = time.time()
        matching = matchDescriptor(listeDesc1, listeDesc2)
        time_matching = time.time()
        print("Matching %s point, took %.3fs " % (matching.size(), time_matching - time_finish))
    else:
        with nogil:
            listeDesc1 = getKeyPoints(img1, octave, interval, l1, verbose)
            listeDesc2 = getKeyPoints(img2, octave, interval, l2, verbose)
            matching = matchDescriptor(listeDesc1, listeDesc2)

    cdef numpy.ndarray[numpy.float32_t, ndim = 2] out = numpy.zeros((matching.size(), 4), dtype="float32")
    get_points(matching, < float *> (out.data))
    del matching, l1, l2, listeDesc1, listeDesc2
    return out

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sift2(img1, img2, bool verbose=False):
    """
    Call SIFT on a pair of images
    @param in1: first image
    @type in1: numpy ndarray
    @param in2: second image
    @type in2: numpy ndarray
    @return: 2D array with n control points and 4 coordinates: in1_0,in1_1,in2_0,in2_1
    """
    cdef size_t i
    cdef float[:, :] data1 = normalize_image(img1)
    cdef float[:, :] data2 = normalize_image(img2)
    cdef keypointslist k1, k2
    cdef siftPar para
    cdef matchingslist matchings
    default_sift_parameters(para)
    if verbose:
        import time
        t0 = time.time()
        compute_sift_keypoints(< float *> & data1[0, 0], k1, data1.shape[1], data1.shape[0], para);
        t1 = time.time()
        print "SIFT took %.3fs image1: %i ctrl points" % (t1 - t0, k1.size())
        t1 = time.time()
        compute_sift_keypoints(< float *> & data2[0, 0], k2, data2.shape[1], data2.shape[0], para);
        t2 = time.time()
        print "SIFT took %.3fs image2: %i ctrl points" % (t2 - t1, k2.size())
        t2 = time.time()
        compute_sift_matches(k1, k2, matchings, para);
        print("Matching: %s point, took %.3fs " % (matchings.size(), time.time() - t2))
    else:
        with nogil:
            compute_sift_keypoints(< float *> & data1[0, 0], k1, data1.shape[1], data1.shape[0], para);
            compute_sift_keypoints(< float *> & data2[0, 0], k2, data2.shape[1], data2.shape[0], para);
            compute_sift_matches(k1, k2, matchings, para);

    cdef numpy.ndarray[numpy.float32_t, ndim = 2] out = numpy.zeros((matchings.size(), 4), dtype="float32")
    for i in range(matchings.size()):
        out[i, 0] = matchings[i].first.y
        out[i, 1] = matchings[i].first.x
        out[i, 2] = matchings[i].second.y
        out[i, 3] = matchings[i].second.x
    return out

def pos(int n, int k, bool vs_first=False):
    """get postion i,j from index k in an upper-filled square array
    [ 0 0 1 2 3 ]
    [ 0 0 4 5 6 ]
    [ 0 0 0 7 8 ]
    [ 0 0 0 0 9 ]
    [ 0 0 0 0 0 ]

    pos(5,9): (3, 4)
    pos(5,8): (2, 4)
    pos(5,7): (2, 3)
    pos(5,6): (1, 4)
    pos(5,5): (1, 3)
    pos(5,4): (1, 2)
    pos(5,3): (0, 4)
    pos(5,2): (0, 3)
    pos(5,1): (0, 2)
    pos(5,0): (0, 1)

    """
    if vs_first:
        return 0, k + 1
    cdef int i, j
    for i in range(n):
        if k < (n - i - 1):
            j = i + 1 + k
            break
        else:
            k = k - (n - i - 1)
    return i, j

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def siftn(*listArg, bool verbose=False, bool vs_first=False):
    """
    Call SIFT on a pair of images
    @param *listArg: images
    @type *listArg: numpy ndarray
    @param verbose: print informations when finished
    @type verbose: boolean
    @param vs_first: calculate sift always vs first img
    @type vs_first: boolean
    @return: 2D array with n control points and 4 coordinates: in1_0,in1_1,in2_0,in2_1
    """
    t0 = time.time()
    cdef int i, j, k, n, m, p, t
    cdef vector[flimage] lstInput
    cdef vector [keypointslist] lstKeypointslist
    cdef vector[matchingslist] lstMatchinglist
    cdef vector[MatchList] lstMatchlist
    cdef vector[float] tmpIdx
    cdef vector [vector[float]] lstIndex
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] tmpNPA
    cdef siftPar para
    cdef int t_value_orsa = 10000
    cdef int verb_value_orsa = 0
    cdef int n_flag_value_orsa = 0
    cdef int mode_value_orsa = 2
    cdef int stop_value_orsa = 0
    cdef float nfa
    cdef MatchList tmpMatchlist
    cdef Match tmpMatch

    default_sift_parameters(para)
    for obj in listArg:
        if isinstance(obj, numpy.ndarray):
            tmpNPA = numpy.ascontiguousarray((255. * (obj.astype("float32") - obj.min()) / < float > (obj.max() - obj.min())).flatten())
            lstInput.push_back(flimage(< int > obj.shape[1], < int > obj.shape[0], < float *> tmpNPA.data))
            lstKeypointslist.push_back(keypointslist())
    n = lstInput.size()
    if vs_first:
        m = n - 1
    else:
        m = n * (n - 1) / 2
    for k in range(m):
        lstMatchinglist.push_back(matchingslist())
        lstMatchlist.push_back(tmpMatchlist)
        lstIndex.push_back(tmpIdx)

    t1 = time.time()
    with nogil:
        for i in prange(n):
            compute_sift_keypoints_flimage(lstInput[i], lstKeypointslist[i], para)
    t2 = time.time()
    with nogil:
        for k in prange(m):
            #Calculate indexes
            if vs_first:
                i = 0
                j = 1 + k
            else:
                t = k
                for i in range(n):
                    if t < (n - i - 1):
                        j = i + 1 + t
                        break
                    else:
                        t = t - (n - i - 1)
            #i,j = pos(n,k)
            compute_sift_matches(lstKeypointslist[i], lstKeypointslist[j], lstMatchinglist[k], para)
    t3 = time.time()
    #with nogil:
    for k in range(m):
            for p in range(< int > lstMatchinglist[k].size()):
                tmpMatch = Match(x1=lstMatchinglist[k][p].first.x,
                                 y1=lstMatchinglist[k][p].first.y,
                                 x2=lstMatchinglist[k][p].second.x,
                                 y2=lstMatchinglist[k][p].second.y)
                lstMatchlist[k].push_back(tmpMatch)
    t4 = time.time()
    with nogil:
        for k in prange(m):
            if (< int > lstMatchinglist[k].size()) > (< int > 20):
                nfa = orsa((lstInput[i].nwidth() + lstInput[j].nwidth()) / 2, (lstInput[i].nheight() + lstInput[j].nheight()) / 2,
                                lstMatchlist[k], lstIndex[k],
                                t_value_orsa, verb_value_orsa, n_flag_value_orsa, mode_value_orsa, stop_value_orsa)
    t5 = time.time()
    out = {}
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] outArray
    for k in range(m):
        tmpMatchlist = lstMatchlist[k]
        if tmpMatchlist.size() == 0:
            out[pos(n, k, vs_first)] = None
        elif tmpMatchlist.size() <= 20:
            outArray = numpy.zeros((tmpMatchlist.size(), 4), dtype="float32")
            for p in range(tmpMatchlist.size()):
                outArray[p, 0] = tmpMatchlist[p].y1
                outArray[p, 1] = tmpMatchlist[p].x1
                outArray[p, 2] = tmpMatchlist[p].y2
                outArray[p, 3] = tmpMatchlist[p].x2
            out[pos(n, k, vs_first)] = outArray
        else:
            outArray = numpy.zeros((lstIndex[k].size(), 4), dtype="float32")
            for p in range(lstIndex[k].size()):
                i = < int > lstIndex[k][p]
                outArray[p, 0] = tmpMatchlist[i].y1
                outArray[p, 1] = tmpMatchlist[i].x1
                outArray[p, 2] = tmpMatchlist[i].y2
                outArray[p, 3] = tmpMatchlist[i].x2
            out[pos(n, k, vs_first)] = outArray
    t6 = time.time()
    print verbose
    if verbose:
        print("Serial setup for SIFT took %.3fs for %i images" % ((t1 - t0), n))
        print("Parallel SIFT took %.3fs for %i images" % ((t2 - t1), n))
        print("Parallel Matching took %.3fs for %i pairs of images" % ((t3 - t2), m))
        print("Serial copy took %.3fs for %i pairs of images" % ((t4 - t3), m))
        print("Parallel ORSA took %.3fs for %i pairs of images" % ((t5 - t4), m))
        print("Serial build of numpy arrays took %.3fs for %i pairs of images" % ((t6 - t5), m))
        for k in range(m):
            print("point %i images pair: %s found %i ctrl pt -> %i" % (k, pos(n, k), lstMatchinglist[k].size(), lstIndex[k].size()))

    return out
#Cython implementaation of fast locking
# author: Stephan Behnel additions from Jerome Kieffer
# http://code.activestate.com/recipes/577336/
# C++ version 
from cpython cimport pythread
from cpython.exc cimport PyErr_NoMemory
from libcpp.list cimport list
from time import time as _time
from time import sleep as _sleep

cdef class FastRLock:
    """Fast, re-entrant locking.

    Under uncongested conditions, the lock is never acquired but only
    counted.  Only when a second thread comes in and notices that the
    lock is needed, it acquires the lock and notifies the first thread
    to release it when it's done.  This is all made possible by the
    wonderful GIL.
    """
    cdef pythread.PyThread_type_lock _real_lock
    cdef long _owner            # ID of thread owning the lock
    cdef int _count             # re-entry count
    cdef int _pending_requests  # number of pending requests for real lock
    cdef bint _is_locked        # whether the real lock is acquired

    def __cinit__(self):
        self._owner = -1
        self._count = 0
        self._is_locked = False
        self._pending_requests = 0
        self._real_lock = pythread.PyThread_allocate_lock()
        if self._real_lock is NULL:
            PyErr_NoMemory()

    def __dealloc__(self):
        if self._real_lock is not NULL:
            pythread.PyThread_free_lock(self._real_lock)
            self._real_lock = NULL

    def acquire(self, bint blocking=True):
        return lock_lock(self, pythread.PyThread_get_thread_ident(), blocking)

    def release(self):
        if self._owner != pythread.PyThread_get_thread_ident():
            raise RuntimeError("cannot release un-acquired lock")
        unlock_lock(self)

    # compatibility with threading.RLock

    def __enter__(self):
        # self.acquire()
        return lock_lock(self, pythread.PyThread_get_thread_ident(), True)

    def __exit__(self, t, v, tb):
        # self.release()
        if self._owner != pythread.PyThread_get_thread_ident():
            raise RuntimeError("cannot release un-acquired lock")
        unlock_lock(self)

    def _is_owned(self):
        return self._owner == pythread.PyThread_get_thread_ident()


cdef inline bint lock_lock(FastRLock lock, long current_thread, bint blocking) nogil:
    # Note that this function *must* hold the GIL when being called.
    # We just use 'nogil' in the signature to make sure that no Python
    # code execution slips in that might free the GIL

    if lock._count:
        # locked! - by myself?
        if current_thread == lock._owner:
            lock._count += 1
            return 1
    elif not lock._pending_requests:
        # not locked, not requested - go!
        lock._owner = current_thread
        lock._count = 1
        return 1
    # need to get the real lock
    return _acquire_lock(
        lock, current_thread,
        pythread.WAIT_LOCK if blocking else pythread.NOWAIT_LOCK)

cdef bint _acquire_lock(FastRLock lock, long current_thread, int wait) nogil:
    # Note that this function *must* hold the GIL when being called.
    # We just use 'nogil' in the signature to make sure that no Python
    # code execution slips in that might free the GIL

    if not lock._is_locked and not lock._pending_requests:
        # someone owns it but didn't acquire the real lock - do that
        # now and tell the owner to release it when done. Note that we
        # do not release the GIL here as we must absolutely be the one
        # who acquires the lock now.
        if not pythread.PyThread_acquire_lock(lock._real_lock, wait):
            return 0
        #assert not lock._is_locked
        lock._is_locked = True
    lock._pending_requests += 1
    with nogil:
        # wait for the lock owning thread to release it
        locked = pythread.PyThread_acquire_lock(lock._real_lock, wait)
    lock._pending_requests -= 1
    #assert not lock._is_locked
    #assert lock._count == 0
    if not locked:
        return 0
    lock._is_locked = True
    lock._owner = current_thread
    lock._count = 1
    return 1

cdef inline void unlock_lock(FastRLock lock) nogil:
    # Note that this function *must* hold the GIL when being called.
    # We just use 'nogil' in the signature to make sure that no Python
    # code execution slips in that might free the GIL

    #assert lock._owner == pythread.PyThread_get_thread_ident()
    #assert lock._count > 0
    lock._count -= 1
    if lock._count == 0:
        lock._owner = -1
        if lock._is_locked:
            pythread.PyThread_release_lock(lock._real_lock)
            lock._is_locked = False


def fimgblur(numpy.ndarray inp not None, float sigma):
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] data = numpy.ascontiguousarray(inp,dtype="float32")
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] output = numpy.empty_like(data)
    cdef int width = data.shape[1], height = data.shape[1]
    imgblur(<float*> data.data,<float*> output.data, width, height, sigma)
    return output
def shrink(numpy.ndarray[numpy.float32_t, ndim = 2] img not None,float factor):
    d0=img.shape[0]
    d1 =img.shape[1] 
    cdef numpy.ndarray[numpy.float32_t, ndim = 2] output = numpy.empty((int(d0/factor),int(d1/factor)),dtype=numpy.float32)
    sample(<float*> & img[0,0],<float*> & output[0,0], factor, d1, d0)
    return output

