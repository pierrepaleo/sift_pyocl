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
 * @param dog_prev: Pointer to global memory with the "previous" difference of gaussians image
 * @param dog: Pointer to global memory with the "current" difference of gaussians image
 * @param dog_next: Pointer to global memory with the "next" difference of gaussians image
 * @param output: Pointer to global memory output *filled with zeros*
 * @param dog_width: integer number of columns of the DOG
 * @param dog_height: integer number of lines of the DOG
 * @param border_dist: integer, distance between inner image and borders (SIFT takes 5)
 * @param peak_thresh: float, threshold (SIFT takes 255.0 * 0.04 / 3.0)
 
 
 notice we still test "dog[pos] > val" to have a simple code. The inequalities have to be strict.
 */


/*
TODO:
-check fabs(val) outside this kernel ? It would avoid the "if"
-confirm usage of fabs instead of fabsf
-confirm the need to return -atan2() rather than atan2 ; to be coherent with python

*/


__kernel void local_maxmin(
	__constant float* dog_prev __attribute__((max_constant_size(MAX_CONST_SIZE))),
	__constant float* dog __attribute__((max_constant_size(MAX_CONST_SIZE))),
	__constant float* dog_next __attribute__((max_constant_size(MAX_CONST_SIZE))),
	__global int* output,
	int dog_width,
	int dog_height,
	int border_dist,
	float peak_thresh)
{

	int gid1 = (int) get_global_id(1);
	int gid0 = (int) get_global_id(0);
	if (gid0 < dog_height && gid1 < dog_width ) {
	
		int res = 0;
		if (gid0 < dog_height - border_dist && gid1 < dog_width - border_dist && gid0 >= border_dist && gid1 >= border_dist) {
	
			float val = dog[gid0*dog_width + gid1];
			//NOTE: "fabsf" instead of "fabs" should be used, for "fabs" if for doubles. Used "fabs" to be coherent with python
			if (fabs(val) > (0.8 * peak_thresh)) {
	
				int c,r,pos;
				int ismax = 0, ismin = 0;
				if (val > 0.0) ismax = 1;
				else ismin = 1;
		
				for (c = gid1 - 1; c <= gid1 + 1; c++) {
					for (r = gid0  - 1; r <= gid0 + 1; r++) {
						pos = r*dog_width + c;
						if (ismax == 1) //if (val > 0.0)
							if (dog_prev[pos] > val || dog[pos] > val || dog_next[pos] > val) ismax = 0;
						if (ismin == 1) //else
							if (dog_prev[pos] < val || dog[pos] < val || dog_next[pos] < val) ismin = 0;
					}
				}
				//these conditions are exclusive
				if (ismax == 1) res = 1; 
				if (ismin == 1) res = -1;	
			} //end greater than threshold		
		} //end "in the inner image"
		output[gid0*dog_width+gid1] = res;
	} //end "in the image"
}








			



