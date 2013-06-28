/*

	Kernels for keypoints processing

	A *group of threads* handles one keypoint, for additional information is required in the keypoint neighborhood

	WARNING: local workgroup size must be at least 128 for orientation_assignment
	
	
	For descriptors (so far) :
	we use shared memory to store temporary 128-histogram (1 per keypoint)
	  therefore, we need 128*N*4 bytes for N keypoints. We have
	  -- 16 KB per multiprocessor for <=1.3 compute capability (GTX <= 295), that allows to process N<=30 keypoints per thread
	  -- 48 KB per multiprocessor for >=2.x compute capability (GTX >= 465, Quadro 4000), that allows to process N<=95 keypoints per thread

*/

typedef float4 keypoint;
#define MIN(i,j) ( (i)<(j) ? (i):(j) )
#define MAX(i,j) ( (i)<(j) ? (j):(i) )
#ifndef WORKGROUP_SIZE
	#define WORKGROUP_SIZE 128
	#define NB_KEYPOINTS_PROCESSED 30 //30*128*4 = 15360 ; also requires 1D kernel (see access to tmp_descriptors) 
#endif



/**
 * \brief Assign an orientation to the keypoints.  This is done by creating a Gaussian weighted histogram
 *   of the gradient directions in the region.  The histogram is smoothed and the largest peak selected.
 *    The results are in the range of -PI to PI.
 *
 * Warning:
 * 			-At this stage, a keypoint is: (peak,r,c,sigma)
 			 After this function, it will be (c,r,sigma,angle)
 *
 * @param keypoints: Pointer to global memory with current keypoints vector.
 * @param grad: Pointer to global memory with gradient norm previously calculated
 * @param ori: Pointer to global memory with gradient orientation previously calculated
 * @param counter: Pointer to global memory with actual number of keypoints previously found
 * @param hist: Pointer to shared memory with histogram (36 values per thread)
 * @param octsize: initially 1 then twiced at each octave
 * @param OriSigma : a SIFT parameter, default is 1.5. Warning : it is not "InitSigma".
 * @param nb_keypoints : maximum number of keypoints
 * @param grad_width: integer number of columns of the gradient
 * @param grad_height: integer num of lines of the gradient
 */


/*

par.OriBins = 36
par.OriHistThresh = 0.8;
-replace "36" by an external paramater ?
-replace "0.8" by an external parameter ?

TODO:
-Memory optimization
	--Use less registers (re-use, calculation instead of assignation)
	--Use local memory for float histogram[36]
-Speed-up
	--Less access to global memory (k.s1 is OK because this is a register)
	--leave the loops as soon as possible
	--Avoid divisions


*/

__kernel void orientation_assignment(
	__global keypoint* keypoints,
	__global float* grad,
	__global float* ori,
	__global int* counter,
	int octsize,
	float OriSigma, //WARNING: (1.5), it is not "InitSigma (=1.6)"
	int nb_keypoints,
	int keypoints_start,
	int keypoints_end,
	int grad_width,
	int grad_height)
{
	int lid0 = get_local_id(0);
	int groupid = get_group_id(0);
	keypoint k = keypoints[groupid];
	if (!(keypoints_start <= groupid && groupid < keypoints_end && k.s1 >=0.0f ))
		return;
	int	bin, prev=0, next=0;
	int old;
	float distsq, gval, angle, interp=0.0;
	float hist_prev,hist_curr,hist_next;
	__local volatile float hist[36];
	__local volatile float hist2[WORKGROUP_SIZE];
	__local volatile int pos[WORKGROUP_SIZE];
	//memset for "pos" and "hist2"
	pos[lid0] = -1;
	hist2[lid0] = 0.0f;

	//local memset
	if (lid0 < 36) hist[lid0]=0.0f;

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
	int i,j,r,c;
	for (r = rmin; r <= rmax; r++) {
		c = cmin + lid0;
		pos[lid0] = -1;
		hist2[lid0] = 0.0f; //do not forget to memset before each re-use...
		if (c <= cmax){
			gval = grad[r*grad_width+c];
			distsq = (r-k.s1)*(r-k.s1) + (c-k.s2)*(c-k.s2);

			if (gval > 0.0f  &&  distsq < (radius*radius) + 0.5f) {
				// Ori is in range of -PI to PI.
				angle = ori[r*grad_width+c];
				bin = (int) (36 * (angle + M_PI_F + 0.001f) / (2.0f * M_PI_F)); //why this offset ?
				//bin = (int) (18.0f * (angle + M_PI_F ) *  M_1_PI_F);
				if (bin >= 0 && bin <= 36) {
					bin = MIN(bin, 35);
					hist2[lid0] = exp(- distsq / (2.0f*sigma*sigma)) * gval;
					pos[lid0] = bin;
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		//We do not have atomic operations on floats...
		if (lid0 == 0) { //this has to be done here ! if not, pos[] is erased !
			for (i=0; i < WORKGROUP_SIZE; i++)
				if (pos[i] != -1) hist[pos[i]] += hist2[i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	
	
	
	/*
		Apply smoothing 6 times for accurate Gaussian approximation
	*/
	for (j=0; j<6; j++) {
		if (lid0 == 0) {
			hist2[0] = hist[0]; //save unmodified hist
			hist[0] = (hist[35] + hist[0] + hist[1]) / 3.0f;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (0 < lid0 && lid0 < 35) {
			hist2[lid0]=hist[lid0];
			hist[lid0] = (hist2[lid0-1] + hist[lid0] + hist[lid0+1]) / 3.0f;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid0 == 35) {
			hist[35] = (hist2[34] + hist[35] + hist[0]) / 3.0f;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	
	
	/* Find maximum value in histogram */

	float maxval = 0.0f;
	int argmax = 0;
	//	Parallel reduction
	if (lid0<32){
		if (lid0+32<36){
			if (hist[lid0]>hist[lid0+32]){
				hist2[lid0]=hist[lid0];
				pos[lid0] = lid0;
			}else{
				hist2[lid0]=hist[lid0+32];
				pos[lid0] = lid0+32;
			}
		}else{
			hist2[lid0]=hist[lid0];
			pos[lid0] = lid0;
		}
	} //now we have hist2[0..32[ that takes [32..36[ into account
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid0<16){
		if (hist2[lid0+16]>hist2[lid0]){
			hist2[lid0]=hist2[lid0+16];
			pos[lid0] = pos[lid0+16];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid0<8 ){
		if (hist2[lid0+ 8]>hist2[lid0]){
			hist2[lid0]=hist2[lid0+ 8];
			pos[lid0] = pos[lid0+ 8];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid0<04){
		if (hist2[lid0+04]>hist2[lid0]){
			hist2[lid0]=hist2[lid0+04];
			pos[lid0] = pos[lid0+04];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid0<02){
		if (hist2[lid0+02]>hist2[lid0]){
			hist2[lid0]=hist2[lid0+02];
			pos[lid0] = pos[lid0+02];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid0==0){
		if (hist2[1]>hist2[0]){
			hist2[0]=hist2[1];
			pos[0] = pos[1];
		}		
		argmax = pos[0];
		maxval = hist2[0];
		
		
	/*
		This maximum value in the histogram is defined as the orientation of our current keypoint
		NOTE: a "true" keypoint has his coordinates multiplied by "octsize" (cf. SIFT)
	*/	
		prev = (argmax == 0 ? 35 : argmax - 1);
		next = (argmax == 35 ? 0 : argmax + 1);
		hist_prev = hist[prev];
		hist_next = hist[next];
		/* //values are positive...
		if (maxval < 0.0f) {
			hist_prev = -hist_prev; //do not directly use hist[prev] which is shared
			maxval = -maxval;
			hist_next = -hist_next;
		}
		*/
		interp = 0.5f * (hist_prev - hist_next) / (hist_prev - 2.0f * maxval + hist_next);
		angle = 2.0f * M_PI_F * (argmax + 0.5f + interp) / 36 - M_PI_F;


		k.s0 = k.s2 *octsize; //c
		k.s1 = k.s1 *octsize; //r
		k.s2 = k.s3 *octsize; //sigma
		k.s3 = angle; 		  //angle
		keypoints[groupid] = k;
		
		pos[0] = argmax;
		hist2[0] = maxval;
		hist2[1] = k.s0; hist2[2] = k.s1; hist2[3] = k.s2; hist2[4] = k.s3; //to avoid central memory access below
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	//sync all threads for these values
	k = (float4) (hist2[1], hist2[2], hist2[3], hist2[4]);
	argmax = pos[0];
	maxval = hist2[0];
	
	/*
		An orientation is now assigned to our current keypoint.
		We can create new keypoints of same (x,y,sigma) but a different angle.
		For every local peak in histogram, every peak of value >= 80% of maxval generates a new keypoint
	*/

	if (lid0 < 36 && lid0 != argmax) {
		i = lid0;
		prev = (i == 0 ? 35 : i - 1);
		next = (i == 35 ? 0 : i + 1);
		hist_prev = hist[prev];
		hist_curr = hist[i];
		hist_next = hist[next];
		if (hist_curr > hist_prev  &&  hist_curr > hist_next && hist_curr >= 0.8f * maxval) {
		/* Use parabolic fit to interpolate peak location from 3 samples. */
		/* //all values are positive...
			if (hist_curr < 0.0f) {
				hist_prev = -hist_prev;
				hist_curr = -hist_curr;
				hist_next = -hist_next;
			}
		*/
			interp = 0.5f * (hist_prev - hist_next) / (hist_prev - 2.0f * hist_curr + hist_next);
			angle = (i + 0.5f + interp) / 18.0f;
			if (angle >= 0  &&  angle <= 2) {
				k.s3 = (angle-1.0f)*M_PI_F;
				old  = atomic_inc(counter);
				if (old < nb_keypoints) keypoints[old] = k;
			}
		} //end "val >= 80%*maxval"
	}
}

/*
**
 * \brief Compute a SIFT descriptor for each keypoint.
 *		WARNING: the workgroup size must be at least 128 (128 is fine, this is the descriptor size)
 *
 *
 *
 *
 * Like in sift.cpp, keypoints are eventually cast to 1-byte integer, for memory sake.
 * However, calculations need to be on float, so we need a temporary descriptors vector in shared memory.
 *
 * In the fist step, the neighborhood of the keypoint is rotated by (-k.s3).
 *   This neighborhood is quite huge ([-42,42]^2), therefore there are many access to global memory for each keypoint
      (from 23^2 = 529 [s=1] to 85**2 = 7225 [s=3.9999] !).
 * To speed this up, we consider a (1D) 128-workgroup for coalescing access to global mem :
 *   -One line of 2*42+1=85 points is copied from global mem. to shared mem
 *   -The rotation and the rest of the processing are made on these points ([k.s0-42,k.s0+42])
 *   -For a given workgroup (that handles columns), a loop is made on the lines [k.s1-42,k.s1+42]
 *
 *
 * PROS:
 *   -coalesced memory access
 *   -normalization/cast are actually done in parallel
 *   -have to create *two* tmp_descriptors of size WORKGROUP_SIZE : one for the inner "i" loop, one for accumulating before casting to uint8. However, it still fits in local memory for all workgroup sizes (N*4*2+N < 10K for N <= 1024)
 *
 *
 * CONS:
 *   -heavy kernel launched once per keypoint
 *   
 *
 *
 * @param keypoints: Pointer to global memory with current keypoints vector
 * @param descriptor: Pointer to global memory with the output SIFT descriptor, cast to uint8
 * //@param tmp_descriptor: Pointer to shared memory with temporary computed float descriptors
 * @param grad: Pointer to global memory with gradient norm previously calculated
 * @param oril: Pointer to global memory with gradient orientation previously calculated
 * @param keypoints_start : index start for keypoints
 * @param keypoints_end: end index for keypoints
 * @param grad_width: integer number of columns of the gradient
 * @param grad_height: integer num of lines of the gradient


 WARNING: scale, row and col must be processed without multiplication by "octsize" !

-par.MagFactor = 3 //"1.5 sigma"
-OriSize  = 8 //number of bins in the local histogram
-par.IndexSigma  = 1.0

 TODO:
-memory optimization

 */


__kernel void descriptor(
	__global keypoint* keypoints,
	__global unsigned char *descriptors,
	//__local float* tmp_descriptors,
	__global float* grad,
	__global float* orim,
	int octsize,
	int keypoints_start,
	int keypoints_end,
	int grad_width,
	int grad_height)
{

	int lid0 = get_local_id(0);
	int groupid = get_group_id(0);
	keypoint k = keypoints[groupid];
	if (!(keypoints_start <= groupid && groupid < keypoints_end && k.s1 >=0.0f))
		return;

	__local volatile float tmp_descriptors[WORKGROUP_SIZE]; //for temporary descriptors, one vector per keypoint i.e per workgroup
	__local volatile float tmp_descriptors_2[WORKGROUP_SIZE];
	__local volatile int pos[WORKGROUP_SIZE]; //for positions (see below)
	//memset
	tmp_descriptors[lid0] = 0.0f;
	tmp_descriptors_2[lid0] = 0.0f;
	pos[lid0] = -1;
	
	int i,j;
	float rx, cx;
	float row = k.s1, col = k.s0, angle = k.s3;
	int	irow = (int) (row + 0.5f), icol = (int) (col + 0.5f);
	float sine = sin((float) angle), cosine = cos((float) angle);

	/* The spacing of index samples in terms of pixels at this scale. */
	float spacing = k.s2 * 3;

	/* Radius of index sample region must extend to diagonal corner of
	index patch plus half sample for interpolation. */
	int iradius = (int) ((1.414f * spacing * 2.5f) + 0.5f);

	/* Examine all points from the gradient image that could lie within the index square. */
	for (i = -iradius; i <= iradius; i++) { //loop on rows
		j = -iradius + lid0; //current column
		pos[lid0] = -1;
		tmp_descriptors[lid0] = 0.0f; //do not forget to memset before each re-use...
		if (j <= iradius) {

			/* Makes a rotation of -angle to achieve invariance to rotation */
			 rx = ((cosine * i - sine * j) - (row - irow)) / spacing + 1.5f;
			 cx = ((sine * i + cosine * j) - (col - icol)) / spacing + 1.5f;

			 /* Compute location of sample in terms of real-valued index array
			 coordinates.  Subtract 0.5 so that rx of 1.0 means to put full
			 weight on index[1] (e.g., when rpos is 0 and 4 is 3. */
			/* Test whether this sample falls within boundary of index patch. */
			if (rx > -1.0f && rx < 4.0f && cx > -1.0f && cx < 4.0f
				 && (irow +i) >= 0  && (irow +i) < grad_height && (icol+j) >= 0 && (icol+j) < grad_width) {
				/* Compute Gaussian weight for sample, as function of radial distance
		 		  from center.  Sigma is relative to half-width of index. */
				float mag = grad[(int)(icol+j) + (int)(irow+i)*grad_width]
							 * exp(- 0.125f*((rx - 1.5f) * (rx - 1.5f) + (cx - 1.5f) * (cx - 1.5f)) );

				/* Subtract keypoint orientation to give ori relative to keypoint. */
				float ori = orim[(int)(icol+j)+(int)(irow+i)*grad_width] -  angle;

				/* Put orientation in range [0, 2*PI]. */
				while (ori > 2.0f*M_PI_F) ori -= 2.0f*M_PI_F;
				while (ori < 0.0f) ori += 2.0f*M_PI_F;

				/* Increment the appropriate locations in the index to incorporate
				 this image sample.  The location of the sample in the index is (rx,cx). */
				int	orr, rindex, cindex, oindex;
				float	rweight, cweight;

				float oval = 4.0f*ori*M_1_PI_F; //8ori/(2pi)

				int	ri = (int)((rx >= 0.0f) ? rx : rx - 1.0f),
					ci = (int)((cx >= 0.0f) ? cx : cx - 1.0f),
					oi = (int)((oval >= 0.0f) ? oval : oval - 1.0f);

				float rfrac = rx - ri,			// Fractional part of location.
					cfrac = cx - ci,
					ofrac = oval - oi;
				/*
				//alternative in OpenCL :
				int ri,ci,oi;
				float	rfrac = fract(rx,&ri),
						cfrac = fract(cx,&ci),
						ofrac = fract(oval,&oi);
				*/
			
				//why are we taking only positive orientations ?
				if (ri >= -1  &&  ri < 4  && oi >=  0  &&  oi <= 8  && rfrac >= 0.0f  &&  rfrac <= 1.0f) {

				/* Put appropriate fraction in each of 8 buckets around this point in the (row,col,ori) dimensions. */
					for (int r = 0; r < 2; r++) {
						rindex = ri + r; 
						if (rindex >=0 && rindex < 4) {
							rweight = mag * ((r == 0) ? 1.0f - rfrac : rfrac);

							for (int c = 0; c < 2; c++) {
								cindex = ci + c;
								if (cindex >=0 && cindex < 4) {
									cweight = rweight * ((c == 0) ? 1.0f - cfrac : cfrac);
									for (orr = 0; orr < 2; orr++) {
										oindex = oi + orr;
										if (oindex >= 8) {  /* Orientation wraps around at PI. */
											oindex = 0;
										}
									/*
										we want descriptors[rindex][cindex][oindex]
											rindex in [0,3]
										 	cindex in [0,3]
										 	oindex in [0,7]
										so	rindex*4 + cindex is in [0,15]
											idx = (rindex*4+cindex)*8 + oindex is in [0,127]
										
										BUT: two different threads can have a same idx (calculated in a non-predictable way),
											 so an assignation here could lead to conflict... 
											 and we do not have atomic operations on floats.
										Solution: a vector of pos[WK_SIZE]
											pos[lid0] = idx; //calculated position relative to the current thread lid0
											Then the thread 0 (no atomic float add) does:
											  descriptors[pos[i]] = (uint8) tmp_descriptors[i] for i in [0,128[
									*/
										pos[lid0] = ((rindex*4 + cindex)*8+oindex); //value in [0,128[
										tmp_descriptors[lid0] = cweight * ((orr == 0) ? 1.0f - ofrac : ofrac);
										tmp_descriptors_2[lid0] = 1; // pos[lid0];
										
									} //end for
								} //end "valid cindex"
							}
						} //end "valid rindex"
					}
				}
			} //end "sample in boundaries"
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid0 == 0) { //this has to be done here ! if not, pos[] is erased !
			for (i=0; i < WORKGROUP_SIZE; i++)//I am lid0, my contribution is tmp_descriptors[lid0] at position pos[lid0]
				if (pos[lid0] != -1) tmp_descriptors_2[pos[i]] =1.0f; //+= tmp_descriptors[i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		
	} //end "i loop"


	/*
		At this point, we have a descriptor associated with our keypoint.
		We have to normalize it, then cast to 1-byte integer
		
		Notes:
		-tmp_descriptors is re-used for communication between threads
		-for tmp_descriptors_2, we use 128 instead of WORKGROUP_SIZE as upper bound,
		   since tmp_descriptors_2 has its values indexed by pos[] whose values are in [0,128[ 
	*/

	// Normalization
	/*
	float norm = 0;
	if (lid0 == 0) { //no float atomic add...
		for (i = 0; i < 128; i++) 
			norm+=pow(tmp_descriptors_2[i],2); //warning: not the same as C "pow"
		norm = rsqrt(norm); //norm = 1.0f/sqrt(norm);
		tmp_descriptors[0] = norm; //broadcast
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	norm = tmp_descriptors[0];
	if (lid0 < 128) {
		tmp_descriptors_2[lid0] *= norm;
	}

	//Threshold to 0.2 of the norm, for invariance to illumination
	//this can be parallelized, but with atomic_add (on descriptors[128*groupid+0] for example), which is slow
	bool changed = false;
	norm = 0;
	if (lid0 == 0) {
		for (i = 0; i < 128; i++) {
			if (tmp_descriptors_2[i] > 0.2f) {
				tmp_descriptors_2[i] = 0.2f;
				changed = true;
			}
			norm += pow(tmp_descriptors_2[i],2);
		}
		tmp_descriptors[1] = (changed == true ? 1.0f : -1.0f); //broadcast
		tmp_descriptors[2] = norm; 
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	norm = tmp_descriptors[2];
	//if values have been changed, we have to normalize again...
	if (tmp_descriptors[1] > 0.0f) {
		norm = rsqrt(norm);
		if (lid0 < 128)
			tmp_descriptors_2[lid0] *= norm;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
*/
	//finally, cast to integer
	//store to global memory : tmp_descriptor_2[lid0] --> descriptors[i][lid0]
	if (lid0 < 128) {
		descriptors[128*groupid+lid0]
			//= (unsigned char) MIN(255,(unsigned char)(512.0f*tmp_descriptors_2[lid0]));
			= (unsigned char) tmp_descriptors_2[lid0];
			
			
	}
}



