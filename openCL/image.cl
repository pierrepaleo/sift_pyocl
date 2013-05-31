/**
 * \brief Gradient of a grayscale image
 *
 * The gradient is computed using central differences in the interior and first differences at the boundaries.
 *
 *
 * @param igray: Pointer to global memory with the input data of the grayscale image
 * @param grad: Pointer to global memory with the output norm of the gradient
 * @param ori: Pointer to global memory with the output orientation of the gradient
 * @param width: integer number of columns of the input image
 * @param height: integer number of lines of the input image
 */
 
#define MAX_CONST_SIZE 16384

__kernel void compute_gradient_orientation(
	__constant float* igray __attribute__((max_constant_size(MAX_CONST_SIZE))),
	__global float *grad,
	__global float *ori,
	int width,
	int height)
{
    
	int gid1 = (int) get_global_id(1);
	int gid0 = (int) get_global_id(0);

	if (gid0 < height && gid1 < width) {
	
		float xgrad, ygrad;
		int pos = gid0*width+gid1;
		
        if (gid1 == 0)
			xgrad = 2.0 * (igray[pos+1] - igray[pos]);
        else if (gid1 == width-1)
			xgrad = 2.0 * (igray[pos] - igray[pos-1]);
        else
			xgrad = igray[pos+1] - igray[pos-1];
        if (gid0 == 0)
			ygrad = 2.0 * (igray[pos] - igray[pos + width]);
        else if (gid0 == height-1)
			ygrad = 2.0 * (igray[pos - width] - igray[pos]);
        else
			ygrad = igray[pos - width] - igray[pos + width];
        

        grad[pos] = sqrt((xgrad * xgrad + ygrad * ygrad))/2;
        ori[pos] = atan2 (-ygrad,xgrad);
      
      }
}





/**
 * \brief Local minimum or maximum detection in a 3x3 neighborhood in 3 DOG
 *
 * output[i,j] = 1 iff [i,j] is a local (3x3) maximum in the 3 DOG
 * output[i,j] = -1 iff [i,j] is a local (3x3) minimum in the 3 DOG
 * output[i,j] = 0 by default (neither maximum nor minimum)
 *
 * @param val: integer to be tested
 * @param dog_prev: Pointer to global memory with the "previous" difference of gaussians image
 * @param dog: Pointer to global memory with the "current" difference of gaussians image
 * @param dog_next: Pointer to global memory with the "next" difference of gaussians image
 * @param output: Pointer to global memory output *filled with zeros*
 * @param dog_width: integer number of columns of the DOG
 * @param dog_height: integer number of lines of the DOG
 */


/*
TODO:
-replace following "define"s by external parameters

*/

#define BORDER_DIST 5
#define PEAKTHRESH 255.0 * 0.04 / 3.0


__kernel void local_maxmin(
	float val,
	__constant float* dog_prev __attribute__((max_constant_size(MAX_CONST_SIZE))),
	__constant float* dog __attribute__((max_constant_size(MAX_CONST_SIZE))),
	__constant float* dog_next __attribute__((max_constant_size(MAX_CONST_SIZE))),
	__global int* output
	int dog_width,
	int dog_height,
	int y0,
	int x0)
{

	int gid1 = (int) get_global_id(1);
	int gid0 = (int) get_global_id(0);

	if (gid0 < dog_height - BORDER_DIST && gid1 < dog_width - BORDER_DIST && gid0 >= BORDER_DIST && gid1 >= BORDER_DIST)
	
	
	float val = dog[gid0*dog_with + gid1];
	if (fabsf(val) > 0.8 * PEAKTHRESH) {//use "fabsf" for floats, for "fabs" if for doubles
	
		int c,r;
		int ismax = 0, ismin = 0;
		if (val > 0.0) ismax = 1;
		else ismin = 1;
		
		for (c = gid1 - 1; x <= gid1 + 1; c++) {
			for (r = gid0  - 1; r <= gid0 + 1; r++) {
				pos = r*dog_width + c;
				if (ismax == 1) //if (val > 0.0)
					if (dog_prev[pos] > val || dog[pos] > val dog_next[pos] > val) ismax = 0;
				if (ismin == 1) //else
					if (dog[pos] < val || dog[pos] < val || dog[pos] < val) ismin = 0;
			}
		}
		
		int res;
		//these conditions are exclusive
		if (ismax == 1) res = 1; 
		if (ismin == 1) res = -1;
		output[pos] = res;
		
	}
}









			



