#!/usr/bin/python
import pyopencl,pyopencl.array
import numpy
ctx = pyopencl.create_some_context()
queue = pyopencl.CommandQueue(ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
x,y,z = numpy.ogrid[-10:10:0.1,-10:10:0.1,-10:10:0.1]
r=numpy.sqrt(x*x+y*y+z*z)
data = ((x * x - y * y + z * z) * numpy.exp(-r)).astype("float32")
gpu_vol = pyopencl.image_from_array(ctx, data, 1)
shape = (200,200)
img = numpy.empty(shape,dtype=numpy.float32)
gpu_img = pyopencl.array.empty(queue, shape, numpy.float32)
prg = open("interpolation.cl").read()
sampler = pyopencl.Sampler(ctx,
                           True, # normalized coordinates
                           pyopencl.addressing_mode.CLAMP_TO_EDGE,
                           pyopencl.filter_mode.LINEAR)

prg = pyopencl.Program(ctx, prg).build()
n = pyopencl.array.to_device(queue, numpy.array([1, 1, 1], dtype=numpy.float32))
c = pyopencl.array.to_device(queue, numpy.array([0.5, 0.5, 0.5], dtype=numpy.float32))
prg.interpolate(queue, (256, 256), (16, 16), gpu_vol, sampler, gpu_img.data,
                numpy.int32(200), numpy.int32(200), c.data, n.data)
img = gpu_img.get()


#timing:
evt = []
evt.append(pyopencl.enqueue_copy(queue, n.data, (2.0*numpy.random.random(3)-1).astype(numpy.float32)))
evt.append(pyopencl.enqueue_copy(queue, c.data, numpy.random.random(3).astype(numpy.float32)))
evt.append(prg.interpolate(queue, (256, 256), (16, 16), gpu_vol, sampler, gpu_img.data,
                numpy.int32(shape[1]), numpy.int32(shape[0]), c.data, n.data))
evt.append(pyopencl.enqueue_copy(queue, img, gpu_img.data))
print("Timings: %.3fms %.3fms %.3fms %.3fms total: %.3fms" % (1e-6 * (evt[0].profile.end - evt[0].profile.start),
                                1e-6 * (evt[1].profile.end - evt[1].profile.start),
                                1e-6 * (evt[2].profile.end - evt[2].profile.start),
                                1e-6 * (evt[3].profile.end - evt[3].profile.start),
                                1e-6 * (evt[-1].profile.end - evt[0].profile.start)))


from pylab import *
imshow(img)
show()
