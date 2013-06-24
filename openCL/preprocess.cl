/*
 *   Project: SIFT: An algorithm for image alignement
 *            Kernel for image pre-processing: Normalization, ...
 *
 *
 *   Copyright (C) 2013 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *   All rights reserved.
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 30/05/2013
 *
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTORS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 */


//OpenCL extensions are silently defined by opencl compiler at compile-time:
#ifdef cl_amd_printf
  #pragma OPENCL EXTENSION cl_amd_printf : enable
  //#define printf(...)
#elif defined(cl_intel_printf)
  #pragma OPENCL EXTENSION cl_intel_printf : enable
#else
  #define printf(...)
#endif
#pragma OPENCL EXTENSION all : enable
//#pragma OPENCL EXTENSION cl_khr_local_float32_base_atomics : enable

//#ifndef WORKGROUP_SIZE
#define WORKGROUP_SIZE 1024
//#endif
    
#define MAX_CONST_SIZE 16384


/**
 * \brief Cast values of an array of uint8 into a float output array.
 *
 * @param array_int:     Pointer to global memory with the input data as unsigned8 array
 * @param array_float:  Pointer to global memory with the output data as float array
 * @param IMAGE_W:        Width of the image
 * @param IMAGE_H:         Height of the image
 */
__kernel void
u8_to_float( __global unsigned char  *array_int,
             __global float *array_float,
             const int IMAGE_W,
             const int IMAGE_H
)
{
    int i = get_global_id(0) * IMAGE_W + get_global_id(1);
    //Global memory guard for padding
    if(i < IMAGE_W*IMAGE_H)
        array_float[i]=(float)array_int[i];
}//end kernel

/**
 * \brief cast values of an array of uint16 into a float output array.
 *
 * @param array_int:    Pointer to global memory with the input data as unsigned16 array
 * @param array_float:  Pointer to global memory with the output data as float array
 * @param IMAGE_W:        Width of the image
 * @param IMAGE_H:         Height of the image
 */
__kernel void
u16_to_float(__global unsigned short  *array_int,
             __global float *array_float,
             const int IMAGE_W,
             const int IMAGE_H
)
{
    int i = get_global_id(0) * IMAGE_W + get_global_id(1);
    //Global memory guard for padding
    if(i < IMAGE_W*IMAGE_H)
        array_float[i]=(float)array_int[i];
}//end kernel


/**
 * \brief convert values of an array of int32 into a float output array.
 *
 * @param array_int:    Pointer to global memory with the data in int
 * @param array_float:  Pointer to global memory with the data in float
 * @param IMAGE_W:        Width of the image
 * @param IMAGE_H:         Height of the image
 */
__kernel void
s32_to_float(    __global int  *array_int,
                __global float  *array_float,
                 const int IMAGE_W,
                 const int IMAGE_H
)
{
    int i = get_global_id(0) * IMAGE_W + get_global_id(1);
    //Global memory guard for padding
    if(i < IMAGE_W*IMAGE_H)
        array_float[i] = (float)(array_int[i]);
}//end kernel

/**
 * \brief convert values of an array of int64 into a float output array.
 *
 * @param array_int:    Pointer to global memory with the data in int
 * @param array_float:  Pointer to global memory with the data in float
 * @param IMAGE_W:        Width of the image
 * @param IMAGE_H:         Height of the image
 */
__kernel void
s64_to_float(    __global long *array_int,
                __global float  *array_float,
                 const int IMAGE_W,
                 const int IMAGE_H
)
{
    int i = get_global_id(0) * IMAGE_W + get_global_id(1);
    //Global memory guard for padding
    if(i < IMAGE_W*IMAGE_H)
        array_float[i] = (float)(array_int[i]);
}//end kernel

/**
 * \brief convert values of an array of float64 into a float output array.
 *
 * @param array_int:    Pointer to global memory with the data in double
 * @param array_float:  Pointer to global memory with the data in float
 * @param IMAGE_W:        Width of the image
 * @param IMAGE_H:         Height of the image
 *
 * COMMENTED OUT AS THIS RUNS ONLY ON GPU WITH FP64
 */
//__kernel void
//double_to_float(__global double *array_int,
//                __global float  *array_float,
//                 const int IMAGE_W,
//                 const int IMAGE_H
//)
//{
//    int i = get_global_id(0) * IMAGE_W + get_global_id(1);
//    //Global memory guard for padding
//    if(i < IMAGE_W*IMAGE_H)
//        array_float[i] = (float)(array_int[i]);
//}//end kernel


/**
 * \brief convert RGB of an array of 3xuint8 into a float output array.
 *
 * @param array_int:    Pointer to global memory with the data in int
 * @param array_float:  Pointer to global memory with the data in float
 * @param IMAGE_W:        Width of the image
 * @param IMAGE_H:         Height of the image
 *
 * WARNING: still untested (formula is the same as PIL)
 */
__kernel void
rgb_to_float(    __global unsigned char *array_int,
                __global float  *array_float,
                 const int IMAGE_W,
                 const int IMAGE_H
)
{
    int i = get_global_id(0) * IMAGE_W + get_global_id(1);
    //Global memory guard for padding
    if(i < IMAGE_W*IMAGE_H)
        array_float[i] = 0.299f*array_int[3*i] + 0.587f*array_int[3*i+1] + 0.114f*array_int[3*i+2];
;
}//end kernel


/**
 * \brief Performs normalization of image between 0 and max_out (255) in place.
 *
 *
 * @param image        Float pointer to global memory storing the image.
 * @param min_in:     Minimum value in the input array
 * @param max_in:     Maximum value in the input array
 * @param max_out:     Maximum value in the output array (255 adviced)
 * @param IMAGE_W:    Width of the image
 * @param IMAGE_H:     Height of the image
 *
**/
__kernel void
normalizes(    __global         float     *image,
            __constant         float * min_in __attribute__((max_constant_size(MAX_CONST_SIZE))),
            __constant         float * max_in __attribute__((max_constant_size(MAX_CONST_SIZE))),
            __constant         float * max_out __attribute__((max_constant_size(MAX_CONST_SIZE))),
            const             int IMAGE_W,
            const             int IMAGE_H
)
{
    float data;
    int i = get_global_id(0) * IMAGE_W + get_global_id(1);
    //Global memory guard for padding
    if(i < IMAGE_W*IMAGE_H)
    {
        data = image[i];
        image[i] = max_out[0]*(data-min_in[0])/(max_in[0]-min_in[0]);
    };//end if in IMAGE
};//end kernel

/**
 * \brief shrink: Subsampling of the image_in into a smaller image_out.
 *
 *
 * @param image_in        Float pointer to global memory storing the big image.
 * @param image_ou        Float pointer to global memory storing the small image.
 * @param scale_w:     Minimum value in the input array
 * @param scale_h:     Maximum value in the input array
 * @param IMAGE_W:    Width of the output image
 * @param IMAGE_H:     Height of the output image
 *
**/
__kernel void
shrink(const __global     float     *image_in,
            __global     float     *image_out,
            const             int scale_w,
            const             int scale_h,
            const             int IMAGE_W,
            const             int IMAGE_H
)
{
    int gid0=get_global_id(0), gid1=get_global_id(1);
    int j,i = gid0 * IMAGE_W + gid1;
    //Global memory guard for padding
    if(i < IMAGE_W*IMAGE_H)
    {
        j = gid0*IMAGE_W*scale_w*scale_h + gid1*scale_w;
        image_out[i] = image_in[j];
    };//end if in IMAGE
};//end kernel


/**
 * \brief bin: resampling of the image_in into a smaller image_out with higher dynamics.
 *
 *
 * @param image_in        Float pointer to global memory storing the big image.
 * @param image_ou        Float pointer to global memory storing the small image.
 * @param scale_w:     Minimum value in the input array
 * @param scale_h:     Maximum value in the input array
 * @param binned_width:    Width of the output image
 * @param IMAGE_H:     Height of the output image
 *
 *Nota: this is a 2D kernel.
**/
__kernel void
bin(        const    __global     float     *image_in,
                    __global     float     *image_out,
            const                 int     scale_width,
            const                 int     scale_heigth,
            const                 int     orig_width,
            const                 int     orig_heigth,
            const                 int     binned_width,
            const                 int     binned_heigth
)
{
    int gid0=get_global_id(0), gid1=get_global_id(1);
    int j,i = gid0 * binned_width + gid1;
    int w, h, size_w, size_h, big_h, big_w;
    float data=0.0f;
    //Global memory guard for padding
    if(i < binned_width*binned_heigth){
        size_h = 0;
        for (h=0; h<scale_heigth; h++){
            big_h = gid0 * scale_heigth + h;
            if (big_h < orig_heigth){
                size_h += 1;
                size_w = 0;
                for (w=0; w<scale_width; w++){
                    big_w = gid1*scale_width + w;
                    if (big_w < orig_width){
                        //j = (gid0 * scale_heigth + h) * (binned_width*scale_width) + (gid1*scale_width + w);
                        size_w += 1;
                        j = big_h * (binned_width*scale_width) + big_w;
                        data += image_in[j];
                    }//end test in image horiz
                };//end for horiz
            }//end test in image vert
        };//end for vertical
        image_out[i] = data/size_h/size_w;
    };//end if in IMAGE
};//end kernel

/**
 * \brief gaussian: Initialize a vector with a gaussian function.
 *
 *
 * @param data:        Float pointer to global memory storing the vector.
 * @param sigma:    width of the gaussian
 * @param size:     size of the function
 *
**/

__kernel void
gaussian(            __global     float     *data,
            const                 float     sigma,
            const                 int     SIZE
)
{
    int gid=get_global_id(0);
    if(gid < SIZE){
        float x = ((float)gid - ((float)SIZE - 1.0f)/2.0f) / sigma;
        float y = exp(-x * x / 2.0f);
        data[gid] = y / sigma / sqrt(2.0f * M_PI_F);
    }
}

/**
 * \brief divide_cst: divide a vector by a constant.
 *
 *
 * @param data:     Float pointer to global memory storing the vector.
 * @param value:    calc data/value
 * @param size:     size of the vector
 *
**/

__kernel void
divide_cst(    __global     float     *data,
            __global    float     *value,
            const         int     SIZE
)
{
    int gid=get_global_id(0);
    if(gid < SIZE){
        data[gid] = data[gid] / value[0];
    }
}


