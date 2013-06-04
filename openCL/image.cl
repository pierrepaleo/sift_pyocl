/**
 *
 * Functions on images
 *
 *
*/


/*
 Keypoint structure : (amplitude, row, column, sigma)
 
 k.x == k.s0 : amplitude
 k.y == k.s1 : row
 k.z == k.s2 : column
 k.w == k.s3 : sigma
 
*/
typedef float4 keypoint;



/*
 Do not use __constant memory for large (usual) images
*/
#define MAX_CONST_SIZE 16384


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
 


__kernel void compute_gradient_orientation(
	__global float* igray, // __attribute__((max_constant_size(MAX_CONST_SIZE))),
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
 * output[i,j] != 0 iff [i,j] is a local (3x3) maximum in the 3 DOG
 * output[i,j] = 0 by default (neither maximum nor minimum)
 *
 * Additionally, the extrema must not lie on an edge (test with ratio of the principal curvatures)
 *
 * @param dog_prev: Pointer to global memory with the "previous" difference of gaussians image
 * @param dog: Pointer to global memory with the "current" difference of gaussians image
 * @param dog_next: Pointer to global memory with the "next" difference of gaussians image
 * @param output: Pointer to global memory output *filled with zeros*
 * @param octsize: initially 1 then twiced at each new octave
 * @param EdgeThresh0: initial upper limit of the curvatures ratio, to test if the point is on an edge
 * @param EdgeThresh: upper limit of the curvatures ratio, to test if the point is on an edge
 * @param border_dist: integer, distance between inner image and borders (SIFT takes 5)
 * @param peak_thresh: float, threshold (SIFT takes 255.0 * 0.04 / 3.0)
 * @param dog_width: integer number of columns of the DOG
 * @param dog_height: integer number of lines of the DOG
 
 notice we still test "dog[pos] > val" to have a simple code. The inequalities have to be strict.
*/


/*
TODO:
-check fabs(val) outside this kernel ? It would avoid the "if"
-confirm usage of fabs instead of fabsf
-confirm the need to return -atan2() rather than atan2 ; to be coherent with python

*/


__kernel void local_maxmin(
	__global float* dog_prev,// __attribute__((max_constant_size(MAX_CONST_SIZE))),
	__global float* dog,// __attribute__((max_constant_size(MAX_CONST_SIZE))),
	__global float* dog_next,// __attribute__((max_constant_size(MAX_CONST_SIZE))),
	__global float* output,
	int border_dist,
	float peak_thresh,
	int octsize,
	float EdgeThresh0,
	float EdgeThresh,
	int dog_width,
	int dog_height)
{

	int gid1 = (int) get_global_id(1);
	int gid0 = (int) get_global_id(0);
	if (gid0 < dog_height && gid1 < dog_width ) {
	
		float res = 0.0f;
		if (gid0 < dog_height - border_dist && gid1 < dog_width - border_dist && gid0 >= border_dist && gid1 >= border_dist) {
	
			float val = dog[gid0*dog_width + gid1];
			
			/*
			The following condition is part of the keypoints refinement: we eliminate the low-contrast points
			NOTE: "fabsf" instead of "fabs" should be used, for "fabs" if for doubles. Used "fabs" to be coherent with python
			*/
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
				/*
				//these conditions are exclusive
				if (ismax == 1) res = 1; 
				if (ismin == 1) res = -1;	
				*/
				
				if (ismax == 1 || ismin == 1) res = val;
				
				/*
				 At this point, we know if "val" is a local extremum or not
				 We have to test if this value lies on an edge (keypoints refinement)
				  This is done by testing the ratio of the principal curvatures, given by the product and the sum of the
				   Hessian eigenvalues
				*/
					
				pos = gid0*dog_width+gid1;
				
				float H00 = dog[(gid0-1)*dog_width+gid1] - 2.0 * dog[pos] + dog[(gid0+1)*dog_width+gid1],
				H11 = dog[pos-1] - 2.0 * dog[pos] + dog[pos+1],
				H01 = ( (dog[(gid0+1)*dog_width+gid1+1] 
						- dog[(gid0+1)*dog_width+gid1-1]) 
						- (dog[(gid0-1)*dog_width+gid1+1] - dog[(gid0-1)*dog_width+gid1-1])) / 4.0;
				
				float det = H00 * H11 - H01 * H01, trace = H00 + H11;

				/*
				   If (trace^2)/det < thresh, the Keypoint is OK.
				   Note that the following "EdgeThresh" seem to be the inverse of the ratio upper limit
				*/

				float edthresh = (octsize <= 1 ? EdgeThresh0 : EdgeThresh);
				
				if (det < edthresh * trace * trace)
					res = 0.0f;
								
				
			}		
		}
		output[gid0*dog_width+gid1] = res;
	}
}




/**
 * \brief Creates keypoints from the matrix returned above. The resulting keypoints vector is for ONE DoG ! 
 *
 *  The matrix returned by local_maxmin has mainly zeros. Working directly on this matrix would be inefficient on GPU, 
 *   so we have to create a keypoints vector from this matrix. 
 * At this stage, the keypoints are detected for a given DoG of std "s", so we shall initialize keypoint.sigma to "s".
 *  The further 3D-interpolation will certainly find keypoints with same (x,y) and different sigma.
 *
 *
 * Here the output is a 1D vector: his size differs from the input matrix. Therefore, we need a counter for this vector.
 * The counter is shared between the GPU threads, so we need an atomic function to avoid conflicts.
 *
 * @param peaks: Pointer to global memory with the output of the previous "local_maxmin" function
 * @param sigma: float "standard deviation" (scale factor) indexing the DoG
 * @param output: Pointer to global memory with the list of keypoints
 * @param nb_keypoints: integer, number of elements of output
 * @param counter: pointer to the shared counter between threads
 * @param width: integer number of columns of "peaks"
 * @param height: integer number of lines of "peaks"
 
 */



__kernel void create_keypoints(
	__global float* peaks,
	float scale,
	__global keypoint* output,
	int nb_keypoints,
	__global int* counter,
	int width,
	int height)
{
	int gid1 = (int) get_global_id(1);
	int gid0 = (int) get_global_id(0);
	if (gid0 < height && gid1 < width ) {
	
		float val = peaks[gid0*width+gid1];
		if (val != 0) {
			int old = atomic_inc(counter);
			keypoint k = 0.0; //no malloc, for this is a float4
			k.s0 = val;
			k.s1 = (float) gid0;
			k.s2 = (float) gid1;
			k.s3 = scale;
			if (old < nb_keypoints) output[old]=k;
		}
	}
}






























































			



