#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import numpy
from pycuda import driver, compiler, gpuarray, tools

# -- initialize the device
import pycuda.autoinit
import scipy.misc

kernel_global = """
__global__ void conv_global(float *inp, float *out, int width, int height)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    uint xplus, yplus;
    if (x<width-1) xplus=x+1;
    else           xplus=x;
    if (y<height-1)yplus=y+1;
    else           yplus=y;


    out[y*width+x] = 0.25f*(inp[y*width+x]+
                            inp[yplus*width+x]+
                            inp[y*width+xplus]+
                            inp[yplus*width+xplus]);
}


__global__ void conv_shared(float *inp, float *out, int width, int height)
{
    const uint shmWidth = blockDim.x + 1;
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float shm[%s];

    shm[ shmWidth*threadIdx.y+threadIdx.x ] = inp[ y*width+x ];
    if (threadIdx.x==blockDim.x-1){
        if (x == width-1)
            shm[ shmWidth*threadIdx.y+threadIdx.x+1 ] = shm[ shmWidth*threadIdx.y+threadIdx.x ];
        else
            shm[ shmWidth*threadIdx.y+threadIdx.x+1 ] = inp[ y*width+x+1 ];
        }
    if (threadIdx.y==blockDim.y-1){
        if (y == height-1)
            shm[ shmWidth*(threadIdx.y+1)+threadIdx.x ] = shm[ shmWidth*threadIdx.y+threadIdx.x ];
        else
            shm[ shmWidth*(threadIdx.y+1)+threadIdx.x ] = inp[ (y+1)*width+x ];
        }
    if ((threadIdx.x==blockDim.x-1)&&(threadIdx.y==blockDim.y-1)){
        if ((x == width-1) && (y == height-1))
            shm[ shmWidth*(threadIdx.y+1)+threadIdx.x+1 ] = shm[ shmWidth*threadIdx.y+threadIdx.x ];
        else
            shm[ shmWidth*(threadIdx.y+1)+threadIdx.x+1 ] = inp[ (y+1)*width+x+1 ];
        }
    __syncthreads();

    out[y*width+x] = 0.25f*(shm[threadIdx.y*shmWidth+threadIdx.x]+
                            shm[(threadIdx.y+1)*shmWidth+threadIdx.x]+
                            shm[threadIdx.y*shmWidth+threadIdx.x+1]+
                            shm[(threadIdx.y+1)*shmWidth+threadIdx.x+1]);
}

"""



BS = 16
# create two random square matrices
a_cpu = scipy.misc.lena().astype("float32")
height, width = a_cpu.shape

aplus = numpy.empty((height+1,width+1),numpy.float32)
aplus[1:,1:]=a_cpu
aplus[:-1,:-1]=a_cpu
ref = 0.25 * (aplus[:-1, :-1] + aplus[1:, :-1] + aplus[:-1, 1:] + aplus[1:, 1:])

# transfer host (CPU) memory to device (GPU) memory
a_gpu = gpuarray.to_gpu(a_cpu)
b_gpu = gpuarray.empty_like(a_gpu)

kernel_code = kernel_global % ((BS + 1) * (BS + 1))

# compile the kernel code
mod = compiler.SourceModule(kernel_code)

# get the kernel function from the compiled module
conv_global = mod.get_function("conv_global")
conv_shared = mod.get_function("conv_shared")
# call the kernel on the card
t0 = time.time()
conv_global(
    # inputs
    a_gpu, b_gpu,
    numpy.int32(width),
    numpy.int32(height),
    # grid of multiple blocks
    grid=(width / BS, height / BS),
    # block of multiple threads
    block=(BS, BS, 1),
    )
print "Global", 1e6 * (time.time() - t0)
b_cpu = b_gpu.get()
numpy.allclose(ref, b_cpu)

t0 = time.time()
conv_shared(
    # inputs
    a_gpu, b_gpu,
    numpy.int32(width),
    numpy.int32(height),
    # grid of multiple blocks
    grid=(width / BS, height / BS),
    # block of multiple threads
    block=(BS, BS, 1),
    )
print "Shared", 1e6 * (time.time() - t0)
b_cpu = b_gpu.get()
numpy.allclose(ref, b_cpu)
if 1:
    import pylab
    pylab.ion()
    fig = pylab.figure()
    sp1 = fig.add_subplot(221)
    sp2 = fig.add_subplot(222)
    sp3 = fig.add_subplot(223)
    sp4 = fig.add_subplot(224)
    sp1.imshow(a_cpu, cmap="gray", interpolation="nearest")
    sp2.imshow(ref, cmap="gray", interpolation="nearest")
    sp3.imshow(b_cpu, cmap="gray", interpolation="nearest")
    sp4.imshow(ref - b_cpu, cmap="gray", interpolation="nearest")
    fig.show()
    raw_input("enter")

