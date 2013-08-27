



/*
 *
 *	Computes the transformation to correct the image given a set of parameters
 *		[[a b c]
 *		[d e f]]
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


__kernel void transform(
	__global float* image,
	__global float* output,
	__global float3* param1,
	__global float3* param2,
	int image_width,
	int image_height,
	float fill,
	int mode)
{
	int gid0 = get_global_id(0);
	int gid1 = get_global_id(1);
	float3 p1 = *param1, p2  = *param2;
	
	if (!(gid0 < image_width && gid1 < image_height))
		return;
	
	//center is at (W/2,H/2)
	int x = gid0 - (image_width>>1),
		y = gid1 - (image_height>>1);
	
	float tx = dot(p1,(float3) (x,y,1.0f)), //x' = a*x + b*y + c
		ty = dot(p2,(float3) (x,y,1.0f)); //y' = d*x + e*y + f

	tx += (image_width>>1);
	ty += (image_height>>1);
	
	int tx_next = ((int) tx) +1,
		tx_prev = (int) tx,
		ty_next = ((int) ty) +1,
		ty_prev = (int) ty;
	
	float interp = fill;

	if (0 <= tx_prev && tx_next < image_width && 0 <= ty_prev && ty_next < image_height) {
		
		//bilinear interpolation
		float interp1 = ((float) (tx_next - tx))/((float) (tx_next - tx_prev)) * image[ty_prev*image_width+tx_prev]
					  + ((float) (tx - tx_prev))/((float) (tx_next - tx_prev)) * image[ty_prev*image_width+tx_next],
					
			interp2 = ((float) (tx_next - tx))/((float) (tx_next - tx_prev)) * image[ty_next*image_width+tx_prev]
					+ ((float) (tx - tx_prev))/((float) (tx_next - tx_prev)) * image[ty_next*image_width+tx_next];
	
		interp = ((float) (ty_next - ty))/((float) (ty_next - ty_prev)) * interp1
			   + ((float) (ty - ty_prev))/((float) (ty_next - ty_prev)) * interp2;
	
	}

	output[gid1*image_width+gid0] = interp;

}








