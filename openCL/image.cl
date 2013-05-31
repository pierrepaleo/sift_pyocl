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

/*
TODO:
-check if igray can be stored in constant memory
-factorization of gid0*width
-when evaluating grad[pos] and ori[pos], cast ygrad and xgrad into double ?
-divide by 2 before returning ?
*/

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
