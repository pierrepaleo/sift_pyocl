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
 *   Last revision: 28/05/2013
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


#ifdef ENABLE_FP64
//	#pragma OPENCL EXTENSION cl_khr_fp64 : enable
	typedef double bigfloat_t;
#else
//	#pragma OPENCL EXTENSION cl_khr_fp64 : disable
	typedef float bigfloat_t;
#endif

#define GROUP_SIZE BLOCK_SIZE

struct lut_point_t
{
	int idx;
	float coef;
};

/**
 * \brief cast values of an array of uint8 into a float output array.
 *
 * @param array_u8: Pointer to global memory with the input data as unsigned8 array
 * @param array_float:  Pointer to global memory with the output data as float array
 */
__kernel void
u8_to_float(__global unsigned char  *array_u8,
		     __global float *array_float
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if(i < NIMAGE)
	array_float[i]=(float)array_u8[i];
}

/**
 * \brief cast values of an array of uint16 into a float output array.
 *
 * @param array_u16: Pointer to global memory with the input data as unsigned16 array
 * @param array_float:  Pointer to global memory with the output data as float array
 */
__kernel void
u16_to_float(__global unsigned short  *array_u16,
		     __global float *array_float
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if(i < NIMAGE)
	array_float[i]=(float)array_u16[i];
}


/**
 * \brief convert values of an array of int32 into a float output array.
 *
 * @param array_int:  Pointer to global memory with the data in int
 * @param array_float:  Pointer to global memory with the data in float
 */
__kernel void
s32_to_float(	__global int  *array_int,
				__global float  *array_float
		)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if(i < NIMAGE)
	array_float[i] = (float)(array_int[i]);
}




/**
 * \brief Performs normalization of image between 0 and 255.
 *
 *
 * @param image	          Float pointer to global memory storing the image.
 *
**/
__kernel void
corrections( 		__global float 	*image,
			const			 float	min,
			const 			 float 	max,
			const			 float	max_out,
			)
{
	float data;
	int i= get_global_id(0);
	if(i < NIMAGE)
	{
		data = image[i];
		image[i] = max_out*(data-min)/(max-min);
	};//end if NIMAGE
};//end kernel

