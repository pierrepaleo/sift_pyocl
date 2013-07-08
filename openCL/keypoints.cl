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

//	Process only valid points
	if ((groupid< keypoints_start) || (groupid >= keypoints_end))
		return;
	keypoint k = keypoints[groupid];
	if (k.s1 < 0.0f )
		return;

	int	bin, prev=0, next=0;
	int old;
	float distsq, gval, angle, interp=0.0;
	float hist_prev,hist_curr,hist_next;
	__local float hist[WORKGROUP_SIZE];
	__local float hist2[WORKGROUP_SIZE];
	__local int pos[WORKGROUP_SIZE];
	float prev2,temp2;
	float ONE_3 = 1.0f / 3.0f;
	float ONE_18 = 1.0f / 18.0f;
	//memset for "pos" and "hist2"
	pos[lid0] = -1;
	hist2[lid0] = 0.0f;
	hist[lid0] = 0.0f;

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

		//memset for "pos" and "hist2"
		pos[lid0] = -1;
		hist2[lid0] = 0.0f;

		c = cmin + lid0;
		pos[lid0] = -1;
		hist2[lid0] = 0.0f; //do not forget to memset before each re-use...
		if (c <= cmax){
			gval = grad[r*grad_width+c];
			distsq = (r-k.s1)*(r-k.s1) + (c-k.s2)*(c-k.s2);
			if (gval > 0.0f  &&  distsq < (radius*radius) + 0.5f) {
				// Ori is in range of -PI to PI.
				angle = ori[r*grad_width+c];
				//bin = (int) (36 * (angle + M_PI_F + 0.001f) / (2.0f * M_PI_F)); //why this offset ?
				bin = (int) (18.0f * (angle + M_PI_F ) *  M_1_PI_F);
				if (bin<0) bin+=36;
				if (bin>35) bin-=36;
				hist2[lid0] = exp(- distsq / (2.0f*sigma*sigma)) * gval;
				pos[lid0] = bin;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		//We are missing atomic operations on floats in OpenCL...
		if (lid0 == 0) { //this has to be done here ! if not, pos[] is erased !
			for (i=0; i < WORKGROUP_SIZE; i++)
				if (pos[i] != -1) hist[pos[i]] += hist2[i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}



//		Apply smoothing 6 times for accurate Gaussian approximation


	for (j=0; j<6; j++) {
		if (lid0 == 0) {
			hist2[0] = hist[0]; //save unmodified hist
			hist[0] = (hist[35] + hist[0] + hist[1]) * ONE_3;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (0 < lid0 && lid0 < 35) {
			hist2[lid0]=hist[lid0];
			hist[lid0] = (hist2[lid0-1] + hist[lid0] + hist[lid0+1]) * ONE_3;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid0 == 35) {
			hist[35] = (hist2[34] + hist[35] + hist[0]) * ONE_3;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

/*
//  WHY not working
	for (j=0; j<3; j++) {
		if (lid0 < 36 ) {
			prev = (lid0 == 0 ? 35 : lid0 - 1);
			next = (lid0 == 35 ? 0 : lid0 + 1);
			hist2[lid0] = (hist[prev] + hist[lid0] + hist[next]) * ONE_3;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid0 < 36 ) {
			prev = (lid0 == 0 ? 35 : lid0 - 1);
			next = (lid0 == 35 ? 0 : lid0 + 1);
			hist[lid0] = (hist2[prev] + hist2[lid0] + hist2[next]) * ONE_3;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
*/

	hist2[lid0] = 0.0f;


	/* Find maximum value in histogram */

	float maxval = 0.0f;
	int argmax = 0;
	//memset for "pos" and "hist2"
	pos[lid0] = -1;
	hist2[lid0] = 0.0f;

	//	Parallel reduction
	if (lid0<32){
		if (lid0+32<36){
			if (hist[lid0]>hist[lid0+32]){
				hist2[lid0] = hist[lid0];
				pos[lid0] = lid0;
			}else{
				hist2[lid0] = hist[lid0+32];
				pos[lid0] = lid0+32;
			}
		}else{
			hist2[lid0] = hist[lid0];
			pos[lid0] = lid0;
		}
	} //now we have hist2[0..32[ that takes [32..36[ into account
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid0<16){
		if (hist2[lid0+16]>hist2[lid0]){
			hist2[lid0] = hist2[lid0+16];
			pos[lid0] = pos[lid0+16];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid0<8 ){
		if (hist2[lid0+ 8]>hist2[lid0]){
			hist2[lid0] = hist2[lid0+ 8];
			pos[lid0] = pos[lid0+ 8];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid0<04){
		if (hist2[lid0+04]>hist2[lid0]){
			hist2[lid0] = hist2[lid0+04];
			pos[lid0] = pos[lid0+04];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid0<02){
		if (hist2[lid0+02]>hist2[lid0]){
			hist2[lid0] = hist2[lid0+02];
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
		angle = (argmax + 0.5f + interp) * ONE_18;
		if (angle<0.0f) angle+=2.0f;
		else if (angle>2.0f) angle-=2.0f;


		k.s0 = k.s2 *octsize; 			//c
		k.s1 = k.s1 *octsize; 			//r
		k.s2 = k.s3 *octsize; 			//sigma
		k.s3 = (angle-1.0f)*M_PI_F; 	//angle
		keypoints[groupid] = k;
//		use local memory to communicate with other threads
		pos[0] = argmax;
		hist2[0] = maxval;
		hist2[1] = k.s0;
		hist2[2] = k.s1;
		hist2[3] = k.s2;
		hist2[4] = k.s3;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	//broadcast these values to all threads
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
			angle = (i + 0.5f + interp) * ONE_18;
			if (angle<0.0f) angle+=2.0f;
			else if (angle>2.0f) angle-=2.0f;
			k.s3 = (angle-1.0f)*M_PI_F;
			old  = atomic_inc(counter);
			if (old < nb_keypoints) keypoints[old] = k;
		} //end "val >= 80%*maxval"
	}
}






/*

	Descriptors... new version with 4x4 workgroup
		
*/
	
	/*
		We have to examine the [-iradius,iradius]^2 zone, maximum [-43,43]^2
		To the next power of two, this is a 128*128 zone !
		Hence, we cannot use one thread per pixel.
		
		Like in SIFT, we divide the patch in 4x4 subregions, each being handled by one thread.
		This is, one thread handles at most 32x32=1024 pixels
		
		
		For memory, we take 16x16=256 pixels per thread, so we can use a 2D shared memory (32*32*4=4096).
		
		
		WARNING: WORKGROUP SIZE MUST BE (4,4,8)
		
	*/
/*
**
 * \brief Compute a SIFT descriptor for each keypoint.
 *		WARNING: the workgroup size must be at least 128 (128 is fine, this is the descriptor size)
 *		UPDATE: the workgroup size MUST BE EXACTLY 128 for if local mem. size is 16kB (GTX <= 295), due to WORKGROUP_SIZE*8 
 * 		 vectors allocating
 *
 *
 *
 * Like in sift.cpp, keypoints are eventually cast to 1-byte integer, for memory sake.
 * However, calculations need to be on float, so we need a temporary descriptors vector in shared memory.
 *
 * In the fist step, the neighborhood of the keypoint is rotated by (-k.s3).
 *   This neighborhood is quite huge ([-42,42]^2), therefore there are many access to global memory for each keypoint
      (from 23^2 = 529 [s=1] to 85**2 = 7225 [s=3.9999] !).
 * To speed this up, we consider a (1D) 128-workgroup for coalescing access to global mem.
 *
 *
 * PROS:
 *   -coalesced memory access
 *   -normalization/cast are actually done in parallel
 *   -have to create *two* tmp_descriptors : one for the inner "i" loop, one for accumulating before casting to uint8. However, it still fits in local memory for all workgroup sizes (N*4*2+N < 10K for N <= 1024)
 *
 *
 * CONS:
 *   -heavy kernel launched once per keypoint
 *   
 *
 *
 * @param keypoints: Pointer to global memory with current keypoints vector
 * @param descriptor: Pointer to global memory with the output SIFT descriptor, cast to uint8
 * @param grad: Pointer to global memory with gradient norm previously calculated
 * @param orim: Pointer to global memory with gradient orientation previously calculated
 * @param octsize: the size of the current octave (1, 2, 4, 8...)
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
	__global float* grad,
	__global float* orim,
	int octsize,
	int keypoints_start,
	int keypoints_end,
	int grad_width,
	int grad_height)
{

	int lid0 = get_local_id(0);
	int lid1 = get_local_id(1);
	int lid2 = get_local_id(2);
	int lid = (lid0*4+lid1)*8+lid2;
	int groupid = get_group_id(0);
	keypoint k = keypoints[groupid];
	if (!(keypoints_start <= groupid && groupid < keypoints_end && k.s1 >=0.0f))
		return;
		
	int i,j,j2;
	
//	__local volatile float tmp_hist[128]; //4x4x8 : [lid0][lid1][r][c][o] where r,c,o are the most inner loops
//	__local volatile int pos[128];
	//other attempt
//	__local volatile float tmp_hist2[16]; //4x4x8 : [lid0][lid1]
//	__local volatile int pos2[16];
	
	
	__local volatile float histogram[128];
	__local volatile float hist2[128*8];
			

	float rx, cx;
	float row = k.s1/octsize, col = k.s0/octsize, angle = k.s3;
	int	irow = (int) (row + 0.5f), icol = (int) (col + 0.5f);
	float sine = sin((float) angle), cosine = cos((float) angle);
	float spacing = k.s2/octsize * 3.0f;
	int radius = (int) ((1.414f * spacing * 2.5f) + 0.5f);
	
	int imin = -64 +32*lid0,
		jmin = -64 +32*lid1;
	int imax = imin+32,
		jmax = jmin+32;
		
	int low_bound = 8*lid1+32*lid0;
	int up_bound = low_bound+8;
	//memset
	histogram[lid] = 0.0f;
	for (i=0; i < 8; i++) hist2[lid*8+i] = 0.0f;
	
	for (i=imin; i < imax; i++) {
		for (j2=jmin/8; j2 < jmax/8; j2++) {	
			j=j2*8+lid2;
			
			 rx = ((cosine * i - sine * j) - (row - irow)) / spacing + 1.5f;
			 cx = ((sine * i + cosine * j) - (col - icol)) / spacing + 1.5f;
			if ((rx > -1.0f && rx < 4.0f && cx > -1.0f && cx < 4.0f
				 && (irow +i) >= 0  && (irow +i) < grad_height && (icol+j) >= 0 && (icol+j) < grad_width)) {
				
				float mag = grad[icol+j + (irow+i)*grad_width]
							 * exp(- 0.125f*((rx - 1.5f) * (rx - 1.5f) + (cx - 1.5f) * (cx - 1.5f)) );
				float ori = orim[icol+j+(irow+i)*grad_width] -  angle;
				
				
				while (ori > 2.0f*M_PI_F) ori -= 2.0f*M_PI_F;
				while (ori < 0.0f) ori += 2.0f*M_PI_F;
				int	orr, rindex, cindex, oindex;
				float	rweight, cweight;

				float oval = 4.0f*ori*M_1_PI_F; 

				int	ri = (int)((rx >= 0.0f) ? rx : rx - 1.0f),
					ci = (int)((cx >= 0.0f) ? cx : cx - 1.0f),
					oi = (int)((oval >= 0.0f) ? oval : oval - 1.0f);

				float rfrac = rx - ri,	
					cfrac = cx - ci,
					ofrac = oval - oi;
				if ((ri >= -1  &&  ri < 4  && oi >=  0  &&  oi <= 8  && rfrac >= 0.0f  &&  rfrac <= 1.0f)) {
					for (int r = 0; r < 2; r++) {
						rindex = ri + r; 
						if ((rindex >=0 && rindex < 4)) {
							rweight = mag * ((r == 0) ? 1.0f - rfrac : rfrac);

							for (int c = 0; c < 2; c++) {
								cindex = ci + c;
								if ((cindex >=0 && cindex < 4)) {
									cweight = rweight * ((c == 0) ? 1.0f - cfrac : cfrac);
									for (orr = 0; orr < 2; orr++) {
										oindex = oi + orr;
										if (oindex >= 8) {  /* Orientation wraps around at PI. */
											oindex = 0;
										}
										int bin = (rindex*4 + cindex)*8+oindex; //value in [0,128[
										
										hist2[lid2+8*bin] += cweight * ((orr == 0) ? 1.0f - ofrac : ofrac);
										
//										histogram[bin] += cweight * ((orr == 0) ? 1.0f - ofrac : ofrac); //works... conflicts ?
//histogram[lid0][lid1][bin]
//histogram[(lid0*4+lid1)*8+oindex] += 1.0f; // cweight * ((orr == 0) ? 1.0f - ofrac : ofrac); //not working ! (lid0,lid1) is not the position to write at !


									} //end "for orr"
								} //end "valid cindex"
							} //end "for c"
						} //end "valid rindex"
					} //end "for r"
				}
			}//end "in the boundaries"
		} //end j loop
	}//end i loop
	
/*	
barrier(CLK_LOCAL_MEM_FENCE);
if (lid0 == 1 && lid1 == 1) {
	for (i=0; i < 4; i++)
		for (j=0; j < 4; j++)
			for (int u=0; u < 8; u++) {
				int p = pos[(i*4+j)*8+u]; //pos[i][j][u];
				if (p != -1) histogram[p] += tmp_hist[(i*4+j)*8+u];
			}
}
barrier(CLK_LOCAL_MEM_FENCE);			
*/

	barrier(CLK_LOCAL_MEM_FENCE);
	histogram[lid] 
		+= hist2[lid*8]+hist2[lid*8+1]+hist2[lid*8+2]+hist2[lid*8+3]+hist2[lid*8+4]+hist2[lid*8+5]+hist2[lid*8+6]+hist2[lid*8+7];





	barrier(CLK_LOCAL_MEM_FENCE);
	
	//memset of 128 values of hist2 before re-use
	hist2[lid] = histogram[lid]*histogram[lid];
	
	
	// Normalization -- work shared by the 16 threads (8 values per thread)
	float inorm = 0.0f;
	
	
	if (lid < 64) {
		hist2[lid] += hist2[lid+64];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid < 32) {
		hist2[lid] += hist2[lid+32];
	}
	
	
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid < 16) {
		hist2[lid] += hist2[lid+16];
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid < 8) {
		hist2[lid] += hist2[lid+8];
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid < 4) {
		hist2[lid] += hist2[lid+4];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid < 2) {
		hist2[lid] += hist2[lid+2];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid == 0) hist2[0] = rsqrt(hist2[1]+hist2[0]);
	barrier(CLK_LOCAL_MEM_FENCE);
	//now we have hist2[0] = 1/sqrt(sum(hist[i]^2))
	
	histogram[lid] *= hist2[0];
	
	
	
	

	//Threshold to 0.2 of the norm, for invariance to illumination
	__local int changed[1];
	if (lid == 0) changed[0] = 0;
	
	if (histogram[lid] > 0.2f) {
		histogram[lid] = 0.2f;
		atomic_inc(changed);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (changed[0]) { //if values have changed, we have to re-normalize
		hist2[lid] = histogram[lid]*histogram[lid];
		if (lid < 64) {
			hist2[lid] += hist2[lid+64];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid < 32) {
			hist2[lid] += hist2[lid+32];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid < 16) {
			hist2[lid] += hist2[lid+16];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid < 8) {
			hist2[lid] += hist2[lid+8];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid < 4) {
			hist2[lid] += hist2[lid+4];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid < 2) {
			hist2[lid] += hist2[lid+2];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid == 0) hist2[0] = rsqrt(hist2[0]+hist2[1]);
		barrier(CLK_LOCAL_MEM_FENCE);
		histogram[lid] *= hist2[0];
	}

	
	
	
	
	
	
//	if (lid2 == 0) {
//	for (i=low_bound; i < up_bound; i++)
		descriptors[128*groupid+lid]
		= (unsigned char) MIN(255,(unsigned char)(512.0f*histogram[lid]));
		//= (unsigned char) histogram[i];
//	}

//end of parallel version


/*
//serial version -- working
	if (lid0 == 0 && lid1 == 0) {
		for (i = 0; i < 128; i++) 
			norm+=pow(histogram[i],2);
		norm = rsqrt(norm);
		for (i = 0; i < 128; i++) 
			histogram[i] *= norm;
	
	
		//Threshold to 0.2 of the norm, for invariance to illumination
		bool changed = false;
		norm = 0;
		if (lid0 == 0 && lid1 == 0) {
			for (i = 0; i < 128; i++) {
				if (histogram[i] > 0.2f) {
					histogram[i] = 0.2f;
					changed = true;
				}
				norm += pow(histogram[i],2);
			}
		}
		//if values have been changed, we have to normalize again...
		if (changed) {
			norm = rsqrt(norm);
			for (i=0; i < 128; i++)
				histogram[i] *= norm;
		}
	
		//finally, cast to integer
		for (i=0; i < 128; i++)
			descriptors[128*groupid+i]
			= (unsigned char) MIN(255,(unsigned char)(512.0f*histogram[i]));
			//= (unsigned char) histogram[i];
	}
//end of serial version
*/	

	

	
}











