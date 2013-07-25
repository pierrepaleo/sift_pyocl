



/*
 *
 *	Computes the transformation to correct the image given a set of parameters
 *		[[a b c]         
 *		[d e f]]  
 *
 *		= [matrix offset]
 *
 * @param image: Pointer to global memory with the input image
 * @param output: Pointer to global memory with the outpu image
 * @param p1: 3-tuple, part of the transformation parameters ([a b c])
 * @param p2: 3-tuple, part of the transformation parameters ([d e f])
 * @param width: width of the input image
 * @param height: height of the input image
 * @param fill: Default value to fill the image with
 *		
 *
 */

//TODO: do not interpolate at the borders

__kernel void transform(
	__global float* image,
	__global float* output,
	__global float4* matrix,
	__global float2* offset,
	int image_width,
	int image_height,
	int output_width,
	int output_height,
	float fill,
	int mode)
{
	int gid0 = get_global_id(0);
	int gid1 = get_global_id(1);
	float4 mat = *matrix;
	float2 off  = *offset;
	
	if (!(gid0 < output_width && gid1 < output_height))
		return;
	
	int x = gid0,
		y = gid1;
	
	float tx = mat.s0*x+mat.s2*y,
		ty = mat.s1*x+mat.s3*y;

	tx += off.s1;
	ty += off.s0;
	
	int tx_next = ((int) tx) +1,
		tx_prev = (int) tx,
		ty_next = ((int) ty) +1,
		ty_prev = (int) ty;
	
	float interp = fill;
	
	//why this rather than "0 <= tx_prev && 0 <= ty_prev && tx_next < image_width && ty_next < image_height" ?	
	if (0 <= tx && tx_next < image_width && 0 <= ty && ty_next < image_height) {
	
		//bilinear interpolation
		float interp1 = ((float) (tx_next - tx)) * image[ty_prev*image_width+tx_prev]
					  + ((float) (tx - tx_prev)) * image[ty_prev*image_width+tx_next],
					
			interp2 = ((float) (tx_next - tx)) * image[ty_next*image_width+tx_prev]
					+ ((float) (tx - tx_prev)) * image[ty_next*image_width+tx_next];
	
		interp = ((float) (ty_next - ty)) * interp1
			   + ((float) (ty - ty_prev)) * interp2;
	
	}

	output[gid1*output_width+gid0] = interp;

}








