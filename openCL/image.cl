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
 * Refactored to return a vector of keypoints, rather than a matrix filled (almost everywhere) with zeros
 *
 * @param dog_prev: Pointer to global memory with the "previous" difference of gaussians image
 * @param dog: Pointer to global memory with the "current" difference of gaussians image
 * @param dog_next: Pointer to global memory with the "next" difference of gaussians image 
 * @param border_dist: integer, distance between inner image and borders (SIFT takes 5)
 * @param peak_thresh: float, threshold (SIFT takes 255.0 * 0.04 / 3.0)
 * @param output: Pointer to global memory output *filled with zeros*
 * @param octsize: initially 1 then twiced at each new octave
 * @param EdgeThresh0: initial upper limit of the curvatures ratio, to test if the point is on an edge
 * @param EdgeThresh: upper limit of the curvatures ratio, to test if the point is on an edge
 * @param counter: pointer to the current position in keypoints vector -- shared between threads
 * @param nb_keypoints: Maximum number of keypoints: size of the keypoints vector
 * @param scale: the scale in the DoG, i.e the index of the current DoG, cast to a float (this is not the std !)
 * @param dog_width: integer number of columns of the DOG
 * @param dog_height: integer number of lines of the DOG
 
*/


/*
TODO:
-check fabs(val) outside this kernel ? It would avoid the "if"
-confirm usage of fabs instead of fabsf
-confirm the need to return -atan2() rather than atan2 ; to be coherent with python

*/


__kernel void local_maxmin(
	__global float* dog_prev,
	__global float* dog,
	__global float* dog_next,
	__global keypoint* output,
	int border_dist,
	float peak_thresh,
	int octsize,
	float EdgeThresh0,
	float EdgeThresh,
	__global int* counter,
	int nb_keypoints,
	float scale,
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
					
				/*
				 At this stage, res != 0.0f iff the current pixel is a good keypoint
				*/
				if (res != 0.0f) {
					int old = atomic_inc(counter);
					keypoint k = 0.0; //no malloc, for this is a float4
					k.s0 = val;
					k.s1 = (float) gid0;
					k.s2 = (float) gid1;
					k.s3 = scale;
					if (old < nb_keypoints) output[old]=k;
				}
				
			}//end "value >thresh"		
		}//end "in the inner image"
	}//end "in the image"
}





/**
 * \brief From the (temporary) keypoints, create a vector of interpolated keypoints 
 * 			(this is the last step of keypoints refinement)
 *  
 *
 * @param dog_prev: Pointer to global memory with the "previous" difference of gaussians image
 * @param dog: Pointer to global memory with the "current" difference of gaussians image
 * @param dog_next: Pointer to global memory with the "next" difference of gaussians image 
 * @param keypoints: Pointer to global memory with current keypoints vector
 * @param output: Pointer to global memory with keypoints vector that will be processed afterwards
 * @param actual_nb_keypoints: actual number of keypoints previously found, i.e previous "counter" final value
 * @param peak_thresh: we are not counting the interpolated values if below the threshold (par.PeakThresh = 255.0*0.04/3.0)
 * @param s: the scale in the DoG, i.e the index of the current DoG, cast to a float (this is not the std !)
 * @param InitSigma: float "par.InitSigma" in SIFT (1.6 by default)
 * @param width: integer number of columns of the DoG
 * @param height: integer number of lines of the DoG
 */
 
/* 
TODO: 
-Writing output directly into input ? It would avoid to check if we are not creating new keypoint
-Taking "actual_nb_keypoints instead of nb_keypoints would spare some memory, is it worth returning it in the previous func ?

-"Check that no keypoint has been created at this location (to avoid duplicates).  Otherwise, mark this map location"	if (map(c,r) > 0.0) return;
	map(c,r) = 1.0
*/


__kernel void interp_keypoint(
	__global float* dog_prev,
	__global float* dog,
	__global float* dog_next,
	__global keypoint* keypoints,
	__global keypoint* output,
	int actual_nb_keypoints,
	float peak_thresh,
	int s,
	float InitSigma,
	int width,
	int height)
{

	int gid1 = (int) get_global_id(1);
	//int gid0 = (int) get_global_id(0);

	if (gid0 < actual_nb_keypoints) {
	
		keypoint k = keypoints[gid0];
		int r = (int) k.s1;
		int c = (int) k.s2;
		
		//pre-allocating variables before entering into the loop
		float g0, g1, g2, 
			H00, H11, H22, H01, H02, H12, H10, H20, H21, 
			K00, K11, K22, K01, K02, K12, K10, K20, K21,
			solution0, solution1, solution2, det, peakval;
		int pos = r*width+c;
		int loop = 1, moveRemain = 5;
		int newr = r, newc = c;
		
		//this loop replaces the recursive "InterpKeyPoint"
		while (loop == 1) { 

			pos = newr*width+newc;

			//Fill in the values of the gradient from pixel differences
			g0 = (dog_next[pos] - dog_prev[pos]) / 2.0;
			g1 = (dog[(newr+1)*width+newc] - dog[(newr-1)*width+newc]) / 2.0;
			g2 = (dog[pos+1] - dog[pos-1]) / 2.0;

			//Fill in the values of the Hessian from pixel differences
			H00 = dog_prev[pos]   - 2.0 * dog[pos] + dog_next[pos];
			H11 = dog[(newr-1)*width+newc] - 2.0 * dog[pos] + dog[(newr+1)*width+newc];
			H22 = dog[pos-1] - 2.0 * dog[pos] + dog[pos+1];
			
			H01 = ( (dog_next[(newr+1)*width+newc] - dog_next[(newr-1)*width+newc])
					- (dog_prev[(newr+1)*width+newc] - dog_prev[(newr-1)*width+newc])) / 4.0;
						
			H02 = ( (dog_next[pos+1] - dog_next[pos-1])
					-(dog_prev[pos+1] - dog_prev[pos-1])) / 4.0;
						
			H12 = ( (dog[(newr+1)*width+newc+1] - dog[(newr+1)*width+newc-1])
					- (dog[(newr-1)*width+newc+1] - dog[(newr-1)*width+newc-1])) / 4.0;
									
			H10 = H01; H20 = H02; H21 = H12;


			//inversion of the Hessian	: det*K = H^(-1)

			// a_13 (a_21 a_32-a_22 a_31)+a_12 (a_23 a_31-a_21 a_33)+a_11 (a_22 a_33-a_23 a_32)
			det = H02*(H10*H21-H11*H20) + H01*(H12*H20-H10*H22) + H00*(H11*H22-H12-H21); 
			/*
			(a_22 a_33-a_23 a_32 | a_13 a_32-a_12 a_33 | a_12 a_23-a_13 a_22
			 a_23 a_31-a_21 a_33 | a_11 a_33-a_13 a_31 | a_13 a_21-a_11 a_23
			 a_21 a_32-a_22 a_31 | a_12 a_31-a_11 a_32 | a_11 a_22-a_12 a_21)
			*/

			K00 = H11*H22 - H12*H21;
			K01 = H02*H21 - H01*H22;
			K02 = H01*H12 - H02*H11;
			K10 = H12*H20 - H10*H22;
			K11 = H00*H22 - H02*H20;
			K12 = H02*H10 - H00*H12;
			K20 = H10*H21 - H11*H20;
			K21 = H01*H20 - H00*H21;
			K22 = H00*H11 - H01*H10;

			//x = -H^(-1)*g
			solution0 = -(g0*K00 + g1*K01 + g2*K02)/det;
			solution1 = -(g0*K10 + g1*K11 + g2*K12)/det;
			solution2 = -(g0*K20 + g1*K21 + g2*K22)/det;


			//interpolated DoG magnitude at this peak
			peakval = dog[pos] + 0.5 * (solution0*g0+solution1*g1+solution2*g2);
		
		
		/* Move to an adjacent (row,col) location if quadratic interpolation is larger than 0.6 units in some direction. 				The movesRemain counter allows only a fixed number of moves to prevent possibility of infinite loops.
		*/

			if (solution1 > 0.6 && gid0 < height - 3)
				newr++;
			else if (solution1 < -0.6 && r > 3)
				newr--;
			if (solution2 > 0.6 && c < width - 3)
				newc++;
			else if (solution2 < -0.6 && c > 3)
				newc--;

			/*
				Loop test
			*/
			if (movesRemain > 0  &&  (newr != r || newc != c))
				movesRemain--;
			else
				loop = 0;
				
		}//end of the big loop
			

		/* Do not create a keypoint if interpolation still remains far outside expected limits, 
			or if magnitude of peak value is below threshold (i.e., contrast is too low).
		*/
		if (fabs(solution0) < 1.5 && fabs(solution1) < 1.5 && fabs(solution2) < 1.5 && fabs(peakval) > peak_thresh) {
			keypoint ki = 0.0; //float4
			ki.s0 = peakval;
			ki.s1 = k.s1 + solution1;
			ki.s2 = k.s2 + solution2;
			ki.s3 = InitSigma * pow(2.0, (s + solution0) / 3.0); //3.0 is "par.Scales"
			output[gid0]=ki;
			
		}
	
	/*
		Better return here and compute histogram in another kernel
	*/
	}
}























































			



