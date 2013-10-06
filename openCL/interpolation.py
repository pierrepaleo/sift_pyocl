#!/usr/bin/python
import pyopencl,pyopencl.array
import numpy
ctx = pyopencl.create_some_context()
queue=pyopencl.CommandQueue(ctx)
x,y,z = numpy.ogrid[-10:10:0.1,-10:10:0.1,-10:10:0.1]
r=numpy.sqrt(x*x+y*y+z*z)
data = ((x * x - y * y + z * z) * numpy.exp(-r)).astype("float32")
gpu_vol = pyopencl.image_from_array(ctx, data, 1)
gpu_img = pyopencl.array.empty(queue, (200, 200), numpy.float32)
prg = open("interpolation.cl").read()
sampler = pyopencl.Sampler(ctx,
                           False, # Not normalized coordinates
                           pyopencl.addressing_mode.CLAMP_TO_EDGE,
                           pyopencl.filter_mode.LINEAR)

prg = pyopencl.Program(ctx, prg).build()
n = pyopencl.array.to_device(numpy.array([1,0,0],dtype=numpy.float32))
c = pyopencl.array.to_device(numpy.array([100, 100, 100], dtype=numpy.float32))

prg.interpolate(queue, (256, 256), (16, 16), gpu_vol, gpu_img.data,
                numpy.int32(200), numpy.int32(200), n.data, c.data)
img = gpu_img.get()
from pylab import *
imshow(img)
show()
