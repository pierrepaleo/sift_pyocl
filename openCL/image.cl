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
#define MIN(i,j) ( (i)<(j) ? (i):(j) )
#define MAX(i,j) ( (i)<(j) ? (j):(i) )

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
 * \brief Local minimum or maximum detection in scale space
 *
 * IMPORTANT: 
 *	-The output have to be Memset to (-1,-1,-1,-1)
 *	-This kernel must not be launched with s = 0 or s = nb_of_dogs (=4 for SIFT)
 *
 * @param DOGS: Pointer to global memory with ALL the coutiguously pre-allocated Differences of Gaussians
 * @param border_dist: integer, distance between inner image and borders (SIFT takes 5)
 * @param peak_thresh: float, threshold (SIFT takes 255.0 * 0.04 / 3.0)
 * @param output: Pointer to global memory output *filled with (-1,-1,-1,-1)* by default for invalid keypoints
 * @param octsize: initially 1 then twiced at each new octave
 * @param EdgeThresh0: initial upper limit of the curvatures ratio, to test if the point is on an edge
 * @param EdgeThresh: upper limit of the curvatures ratio, to test if the point is on an edge
 * @param counter: pointer to the current position in keypoints vector -- shared between threads
 * @param nb_keypoints: Maximum number of keypoints: size of the keypoints vector
 * @param scale: the scale in the DoG, i.e the index of the current DoG (this is not the std !)
 * @param total_width: integer number of columns of ALL the (contiguous) DOGs. We have total_height = height
 * @param width: integer number of columns of a DOG.
 * @param height: integer number of lines of a DOG
 
*/


/*
TODO:
-check fabs(val) outside this kernel ? It would avoid the "if"
-confirm usage of fabs instead of fabsf
-confirm the need to return -atan2() rather than atan2 ; to be coherent with python

*/


__kernel void local_maxmin(
	__global float* DOGS,
	__global keypoint* output,
	int border_dist,
	float peak_thresh,
	int octsize,
	float EdgeThresh0,
	float EdgeThresh,
	__global int* counter,
	int nb_keypoints,
	int scale,
	int width,
	int height)
{

	int gid1 = (int) get_global_id(1);
	int gid0 = (int) get_global_id(0);
	/*
		As the DOGs are contiguous, we have to test if (gid0,gid1) is actually in DOGs[s]
	*/
	
	if ((gid0 < height - border_dist) && (gid1 < width - border_dist) && (gid0 >= border_dist) && (gid1 >= border_dist)) {
		int index_dog_prev = (scale-1)*(width*height);
		int index_dog =scale*(width*height);
		int index_dog_next =(scale+1)*(width*height);
				
		float res = 0.0f;
		float val = DOGS[index_dog+gid0*width + gid1];
		
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
					pos = r*width + c;
					if (ismax == 1) //if (val > 0.0)
						if (DOGS[index_dog_prev+pos] > val || DOGS[index_dog+pos] > val || DOGS[index_dog_next+pos] > val) ismax = 0;
					if (ismin == 1) //else
						if (DOGS[index_dog_prev+pos] < val || DOGS[index_dog+pos] < val || DOGS[index_dog_next+pos] < val) ismin = 0;
				}
			}
			
			if (ismax == 1 || ismin == 1) res = val;
			
			/*
			 At this point, we know if "val" is a local extremum or not
			 We have to test if this value lies on an edge (keypoints refinement)
			  This is done by testing the ratio of the principal curvatures, given by the product and the sum of the
			   Hessian eigenvalues
			*/
				
			pos = gid0*width+gid1;
			
			float H00 = DOGS[index_dog+(gid0-1)*width+gid1] - 2.0 * DOGS[index_dog+pos] + DOGS[index_dog+(gid0+1)*width+gid1],
			H11 = DOGS[index_dog+pos-1] - 2.0 * DOGS[index_dog+pos] + DOGS[index_dog+pos+1],
			H01 = ( (DOGS[index_dog+(gid0+1)*width+gid1+1] 
					- DOGS[index_dog+(gid0+1)*width+gid1-1]) 
					- (DOGS[index_dog+(gid0-1)*width+gid1+1] - DOGS[index_dog+(gid0-1)*width+gid1-1])) / 4.0;
			
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
				k.s3 = (float) scale;
				if (old < nb_keypoints) output[old]=k;
			}
			
		}//end "value >thresh"		
	}//end "in the inner image"
}





/**
 * \brief From the (temporary) keypoints, create a vector of interpolated keypoints 
 * 			(this is the last step of keypoints refinement)
 *  	 Note that we take the value (-1,-1,-1) for invalid keypoints
 * 
 * @param DOGS: Pointer to global memory with ALL the coutiguously pre-allocated Differences of Gaussians
 * @param keypoints: Pointer to global memory with current keypoints vector. It will be modified with the interpolated points
 * @param actual_nb_keypoints: actual number of keypoints previously found, i.e previous "counter" final value
 * @param peak_thresh: we are not counting the interpolated values if below the threshold (par.PeakThresh = 255.0*0.04/3.0)
 * @param InitSigma: float "par.InitSigma" in SIFT (1.6 by default)
 * @param width: integer number of columns of the DoG
 * @param height: integer number of lines of the DoG
 */
 

__kernel void interp_keypoint(
	__global float* DOGS,
	__global keypoint* keypoints,
	int actual_nb_keypoints,
	float peak_thresh,
	float InitSigma,
	int width,
	int height)
{

	//int gid1 = (int) get_global_id(1);
	int gid0 = (int) get_global_id(0);

	if (gid0 < actual_nb_keypoints) {
		keypoint k = keypoints[gid0];
		int r = (int) k.s1;
		int c = (int) k.s2;
		int scale = (int) k.s3;
		
		int index_dog_prev = (scale-1)*(width*height);
		int index_dog =scale*(width*height);
		int index_dog_next =(scale+1)*(width*height);
		
		
		
		//pre-allocating variables before entering into the loop
		float g0, g1, g2, 
			H00, H11, H22, H01, H02, H12, H10, H20, H21, 
			K00, K11, K22, K01, K02, K12, K10, K20, K21,
			solution0, solution1, solution2, det, peakval;
		int pos = r*width+c;
		int loop = 1, movesRemain = 5;
		int newr = r, newc = c;
		
		//this loop replaces the recursive "InterpKeyPoint"
		while (loop == 1) { 

			pos = newr*width+newc;

			//Fill in the values of the gradient from pixel differences
			g0 = (DOGS[index_dog_next+pos] - DOGS[index_dog_prev+pos]) / 2.0;
			g1 = (DOGS[index_dog+(newr+1)*width+newc] - DOGS[index_dog+(newr-1)*width+newc]) / 2.0;
			g2 = (DOGS[index_dog+pos+1] - DOGS[index_dog+pos-1]) / 2.0;

			//Fill in the values of the Hessian from pixel differences
			H00 = DOGS[index_dog_prev+pos]   - 2.0 * DOGS[index_dog+pos] + DOGS[index_dog_next+pos];
			H11 = DOGS[index_dog+(newr-1)*width+newc] - 2.0 * DOGS[index_dog+pos] + DOGS[index_dog+(newr+1)*width+newc];
			H22 = DOGS[index_dog+pos-1] - 2.0 * DOGS[index_dog+pos] + DOGS[index_dog+pos+1];
			
			H01 = ( (DOGS[index_dog_next+(newr+1)*width+newc] - DOGS[index_dog_next+(newr-1)*width+newc])
					- (DOGS[index_dog_prev+(newr+1)*width+newc] - DOGS[index_dog_prev+(newr-1)*width+newc])) / 4.0;
						
			H02 = ( (DOGS[index_dog_next+pos+1] - DOGS[index_dog_next+pos-1])
					-(DOGS[index_dog_prev+pos+1] - DOGS[index_dog_prev+pos-1])) / 4.0;
						
			H12 = ( (DOGS[index_dog+(newr+1)*width+newc+1] - DOGS[index_dog+(newr+1)*width+newc-1])
					- (DOGS[index_dog+(newr-1)*width+newc+1] - DOGS[index_dog+(newr-1)*width+newc-1])) / 4.0;
									
			H10 = H01; H20 = H02; H21 = H12;


			//inversion of the Hessian	: det*K = H^(-1)
			
			det = -(H02*H11*H20) + H01*H12*H20 + H02*H10*H21 - H00*H12*H21 - H01*H10*H22 + H00*H11*H22;

			K00 = H11*H22 - H12*H21;
			K01 = H02*H21 - H01*H22;
			K02 = H01*H12 - H02*H11;
			K10 = H12*H20 - H10*H22;
			K11 = H00*H22 - H02*H20;
			K12 = H02*H10 - H00*H12;
			K20 = H10*H21 - H11*H20;
			K21 = H01*H20 - H00*H21;
			K22 = H00*H11 - H01*H10;

			/*
				x = -H^(-1)*g 
			 As the Taylor Serie is calcualted around the current keypoint, 
			 the position of the true extremum x_opt is exactly the "offset" between x and x_opt ("x" is the origin)
			*/
			solution0 = -(g0*K00 + g1*K01 + g2*K02)/det; //"offset" in sigma
			solution1 = -(g0*K10 + g1*K11 + g2*K12)/det; //"offset" in r
			solution2 = -(g0*K20 + g1*K21 + g2*K22)/det; //"offset" in c


			//interpolated DoG magnitude at this peak
			peakval = DOGS[index_dog+pos] + 0.5 * (solution0*g0+solution1*g1+solution2*g2);
		
		
		/* Move to an adjacent (row,col) location if quadratic interpolation is larger than 0.6 units in some direction. 				The movesRemain counter allows only a fixed number of moves to prevent possibility of infinite loops.
		*/

			if (solution1 > 0.6 && newr < height - 3)
				newr++; //if the extremum is too far (along "r" here), we get closer if we can
			else if (solution1 < -0.6 && newr > 3)
				newr--;
			if (solution2 > 0.6 && newc < width - 3)
				newc++;
			else if (solution2 < -0.6 && newc > 3)
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
		keypoint ki = 0.0; //float4
		if (fabs(solution0) < 1.5 && fabs(solution1) < 1.5 && fabs(solution2) < 1.5 && fabs(peakval) > peak_thresh) {
			ki.s0 = peakval;
			ki.s1 = k.s1 + solution1;
			ki.s2 = k.s2 + solution2;
			ki.s3 = InitSigma * pow(2.0, (((float) scale) + solution0) / 3.0); //3.0 is "par.Scales"
		}
		else { //the keypoint was not correctly interpolated : we reject it
			ki.s0 = -1.0f; ki.s1 = -1.0f; ki.s2 = -1.0f; ki.s3 = -1.0f;
		}
		
		keypoints[gid0]=ki;	
	
	/*
		Better return here and compute histogram in another kernel
	*/
	}
}








/**
 * \brief Assign an orientation to the keypoints.  This is done by creating a Gaussian weighted histogram
 *   of the gradient directions in the region.  The histogram is smoothed and the largest peak selected.
 *    The results are in the range of -PI to PI.
 *
 * Warning: -this function requires "keypoints" to be compacted
 * 			-At this stage, a keypoint is: (peak,r,c,sigma)
 			 After this function, it will be (c,r,sigma,angle)
 			-The workgroup size have to be "small" in order to achieve "hist[36]"
 *
 * @param keypoints: Pointer to global memory with current keypoints vector. It will be modified with the interpolated points
 * @param grad: Pointer to global memory with gradient norm previously calculated
 * @param ori: Pointer to global memory with gradient orientation previously calculated
 * @param counter: Pointer to global memory with actual number of keypoints previously found
 * @param octsize: initially 1 then twiced at each octave
 * @param OriSigma : a SIFT parameter, default is 1.5. Warning : it is not "InitSigma".
 * @param nb_keypoints : maximum number of keypoints
 * @param grad_width: integer number of columns of the gradient
 * @param grad_height: integer num of lines of the gradient
 */


/*
TODO: use less memory

par.OriBins = 36
par.OriHistThresh = 0.8;
-replace "36" by an external paramater ?
-replace "0.8" by an external parameter ?

*/

__kernel void orientation_assignment(
	__global keypoint* keypoints,
	__global float* grad, 
	__global float* ori,
	__global int* counter,
	int octsize,
	float OriSigma, //WARNING: (1.5), it is not "InitSigma (=1.6)"
	int nb_keypoints,
	int actual_nb_keypoints,
	int grad_width,
	int grad_height)
{
	int gid0 = (int) get_global_id(0);

	if (gid0 < actual_nb_keypoints) { //do not use *counter, for it will be modified below
		keypoint k = keypoints[gid0];
		int	bin, prev, next;
		int old;
		float distsq, gval, weight, angle, interp=0.0;
		//if compiler is GCC, we can use float hist[36] = {[0 ... 35] = 0.0};
		float hist[36] = { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f, 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f, 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f};
		float radius2, sigma2;
		int	row = (int) (k.s1 + 0.5),
			col = (int) (k.s2 + 0.5);

		/* Look at pixels within 3 sigma around the point and sum their
		  Gaussian weighted gradient magnitudes into the histogram. */
		  
		float sigma = OriSigma * k.s3;
		int	radius = (int) (sigma * 3.0);
		int rmin = MAX(0,row - radius);
		int cmin = MAX(0,col - radius);
		int rmax = MIN(row + radius,grad_height - 2);
		int cmax = MIN(col + radius,grad_width - 2);
		radius2 = (float)(radius * radius);
		sigma2 = 2.0f*sigma*sigma;
		for (int r = rmin; r <= rmax; r++) {
			for (int c = cmin; c <= cmax; c++) {

				gval = grad[r*grad_width+c];
				distsq = (r-k.s1)*(r-k.s1) + (c-k.s2)*(c-k.s2);
				
				if (gval > 0.0f  &&  distsq < radius2 + 0.5f) {
					weight = exp(- distsq / sigma2);
					/* Ori is in range of -PI to PI. */
					angle = ori[r*grad_width+c];
					//FIXME: it should be "35" !
					bin = (int) (36 * (angle + M_PI_F + 0.001f) / (2.0f * M_PI_F));
					if (bin >= 0 && bin <= 36) {
						bin = MIN(bin, 36 - 1);
						hist[bin] += weight * gval; //FIXME: numerical error accumulation, consider Kahan summation
						
					}
				}
			}
		}

		/* Apply smoothing 6 times for accurate Gaussian approximation. */
		float prev2, temp2;
		for (int j = 0; j < 6; j++) {
			prev2 = hist[35];
			for (int i = 0; i < 36; i++) {
				temp2 = hist[i];
				hist[i] = ( prev2 + hist[i] + hist[(i + 1 == 36) ? 0 : i + 1] ) / 3.0f;
				prev2 = temp2;
			}
		}
        
		/* Find maximum value in histogram. */
		float maxval = 0.0f;
		int argmax = 0;
		for (int i = 0; i < 36; i++)
			if (hist[i] > maxval) { maxval = hist[i]; argmax = i; }

		/*
			This maximum value in the histogram is defined as the orientation of our current keypoint
			NOTE: a "true" keypoint has his coordinates multiplied by "octsize" (cf. SIFT)
		*/
		prev = (argmax == 0 ? 36 - 1 : argmax - 1);
		next = (argmax == 36 - 1 ? 0 : argmax + 1);
		if (maxval < 0.0f) {
			hist[prev] = -hist[prev]; maxval = -maxval; hist[next] = -hist[next];
		}
		interp = 0.5f * (hist[prev] - hist[next]) / (hist[prev] - 2.0f * maxval + hist[next]);
		angle = 2.0f * M_PI_F * (argmax + 0.5f + interp) / 36 - M_PI_F;
		
		
		k.s0 = octsize * k.s2; //c
		k.s1 = octsize * k.s1; //r
		k.s2 = octsize * k.s3; //sigma
		k.s3 = angle;		   //angle
		
		keypoints[gid0] = k;
		
		/*
			An orientation is now assigned to our current keypoint.
			We can create new keypoints of same (x,y,sigma) but a different angle.
		 	For every local peak in histogram, every peak of value >= 80% of maxval generates a new keypoint	
		*/
		
		keypoint k2 = 0.0; k2.s0 = k.s0; k2.s1 = k.s1; k2.s2 = k.s2;
		for (int i = 0; i < 36; i++) {
			prev = (i == 0 ? 36 -1 : i - 1);
			next = (i == 36 -1 ? 0 : i + 1);
			if (hist[i] > hist[prev]  &&  hist[i] > hist[next] && hist[i] >= 0.8f * maxval && i != argmax) {
				/* Use parabolic fit to interpolate peak location from 3 samples. Set angle in range -PI to PI. */
				if (hist[i] < 0.0f) {
					hist[prev] = -hist[prev]; hist[i] = -hist[i]; hist[next] = -hist[next];
				}
				if (hist[i] >= hist[prev]  &&  hist[i] >= hist[next]) 
		 			interp = 0.5f * (hist[prev] - hist[next]) / (hist[prev] - 2.0f * hist[i] + hist[next]);
			
				angle = 2.0f * M_PI_F * (i + 0.5f + interp) / 36 - M_PI_F;
				if (angle >= -M_PI_F  &&  angle <= M_PI_F) {
					k2.s3 = angle;
					old  = atomic_inc(counter);
					if (old < nb_keypoints) keypoints[old] = k2;
				}
			} //end "val >= 80%*maxval"
		} //end loop in histogram
	} //end "in the vector"
}










































			



