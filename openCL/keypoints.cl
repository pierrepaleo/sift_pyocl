/*

	Kernels for keypoints processing

	A *group of threads* handles one keypoint, for additional information is required in the keypoint neighborhood

	WARNING: local workgroup size must be at least 128 for orientation_assignment

*/

typedef float4 keypoint;
#define MIN(i,j) ( (i)<(j) ? (i):(j) )
#define MAX(i,j) ( (i)<(j) ? (j):(i) )
#ifndef WORKGROUP_SIZE
	#define WORKGROUP_SIZE 128
#endif





/**
 * \brief Compute a SIFT descriptor for each keypoint.
 *
 * Like in sift.cpp, keypoints are eventually cast to 1-byte integer, for memory sake.
 * However, calculations need to be on float, so we need a temporary descriptors vector in shared memory.
 *
 * In this kernel, we launch WG*nb_keypoints threads, so that one threads handles a 128-vector (grad and ori).
 * This enables to make coalescing global memory access for grad and ori.
 * The 128 vector "hist2" is finally send in a 36 vector "hist"
 *
 * @param keypoints: Pointer to global memory with current keypoints vector
 * @param descriptor: Pointer to global memory with the output SIFT descriptor, cast to uint8
 * @param tmp_descriptor: Pointer to shared memory with temporary computed float descriptors
 * @param grad: Pointer to global memory with gradient norm previously calculated
 * @param oril: Pointer to global memory with gradient orientation previously calculated
 * @param keypoints_start : index start for keypoints
 * @param keypoints_end: end index for keypoints
 * @param grad_width: integer number of columns of the gradient
 * @param grad_height: integer num of lines of the gradient


 WARNING: scale, row and col must not have been multiplied by octsize/octscale here !

-par.MagFactor = 3 //"1.5 sigma"
-OriSize  = 8 //number of bins in the local histogram
-par.IndexSigma  = 1.0

 TODO:
-(c,r,sigma) are not multiplied by octsize yet. It can be done in this kernel.
-memory optimization


Vertical keypoints (gid0) :
desc[128*gid0 + i] with i in range(0,128)

Horizontal keypoints (gid1) :
desc[W*i+gid0] with i in range(0,128) and W = keypoints_end-keypoints_start+1

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
	//int gid0 = (int) get_global_id(0);
	int lid0 = (int) get_local_id(0);
	int groupid = get_group_id(0);
	if ((keypoints_start > groupid || groupid >= keypoints_end))
		return;
	keypoint k = keypoints[groupid];
	if  (k.s1 < 0.0f )
		return;
	int	bin, prev, next;
	int old;
	float distsq, gval, angle, interp=0.0;
	float hist_prev,hist_curr,hist_next;
	__local float hist[WORKGROUP_SIZE];
	__local float hist2[WORKGROUP_SIZE];
	__local int pos[WORKGROUP_SIZE];
	float prev2,temp2;
	//local memset
//	if (lid0 < 36) 
	hist[lid0]=0.0f;

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
		if (c <= cmax){
			gval = grad[r*grad_width+c];
			distsq = (r-k.s1)*(r-k.s1) + (c-k.s2)*(c-k.s2);

			if (gval > 0.0f  &&  distsq < ((float) (radius*radius)) + 0.5f) {
				// Ori is in range of -PI to PI.
				angle = ori[r*grad_width+c];
				bin = (int) (18.0f * (angle + M_PI_F ) *  M_1_PI_F);
				if (bin<0) bin+=36;
				else if (bin>35) bin-=36;
				hist2[lid0] = exp(- distsq / (2.0f*sigma*sigma)) * gval;
				pos[lid0] = bin;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		//We are missing atomic operations on floats in OpenCL...

		if (lid0 == 0) {
			for (i=0; i < WORKGROUP_SIZE; i++) {
				if (pos[i] != -1) hist[pos[i]] += hist2[i];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	hist2[lid0] = 0.0f;
	if (lid0 ==0){
		for (j=0;j<6;j++){
			prev2 = hist[35];
			for (i = 0; i < 36; i++) {
				temp2 = hist[i];
				hist[i] = ( prev2 + hist[i] + hist[((i + 1 == 36) ? 0 : i + 1)] ) / 3.0f;
				prev2 = temp2;
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
/*
	barrier(CLK_LOCAL_MEM_FENCE);
	// Apply smoothing 6 times for accurate Gaussian approximation.
	if (lid0<36){
		prev = (lid0 == 0 ? 35 : lid0 - 1);
		next = (lid0 == 35 ? 0 : lid0 + 1);
		hist2[lid0] = (hist[prev]+hist[lid0]+hist[next])/ 3.0f;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	hist[lid0] = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid0<36){
		prev = (lid0 == 0 ? 35 : lid0 - 1);
		next = (lid0 == 35 ? 0 : lid0 + 1);
		hist[lid0] = (hist2[prev]+hist2[lid0]+hist2[next])/ 3.0f;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	hist2[lid0] = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid0<36){
		prev = (lid0 == 0 ? 35 : lid0 - 1);
		next = (lid0 == 35 ? 0 : lid0 + 1);
		hist2[lid0] = (hist[prev]+hist[lid0]+hist[next])/ 3.0f;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	hist[lid0] = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid0<36){
		prev = (lid0 == 0 ? 35 : lid0 - 1);
		next = (lid0 == 35 ? 0 : lid0 + 1);
		hist[lid0] = (hist2[prev]+hist2[lid0]+hist2[next])/ 3.0f;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	hist2[lid0] = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid0<36){
		prev = (lid0 == 0 ? 35 : lid0 - 1);
		next = (lid0 == 35 ? 0 : lid0 + 1);
		hist2[lid0] = (hist[prev]+hist[lid0]+hist[next])/ 3.0f;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	hist[lid0] = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid0<36){
		prev = (lid0 == 0 ? 35 : lid0 - 1);
		next = (lid0 == 35 ? 0 : lid0 + 1);
		hist[lid0] = (hist2[prev]+hist2[lid0]+hist2[next])/ 3.0f;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	hist2[lid0] = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
*/

	/* Find maximum value in histogram: Todo parallel max. */
	float maxval = 0.0f;
	int argmax = 0;
	//memset for "pos" and "hist2"
	pos[lid0] = -1;
	hist2[lid0] = 0.0f;

	/*
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
	}
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
	*/
	if (lid0==0){
		
//		if (hist2[1]>hist2[0]){
//			hist2[0]=hist2[1];
//			pos[0] = pos[1];
//		}
//		argmax = pos[0];
//		maxval = hist2[0];

		argmax = 0;
		maxval = 0.0f;
		
		for (i=0;i<36;i++){
			if (hist[i]>maxval){
				maxval = hist[i];
				argmax=i;
			}
		}
		pos[0] = argmax;
		hist2[0] = maxval;
	//		This maximum value in the histogram is defined as the orientation of our current keypoint
	//		NOTE: a "true" keypoint has his coordinates multiplied by "octsize" (cf. SIFT)
		prev = (argmax == 0 ? 35 : argmax - 1);
		next = (argmax == 35 ? 0 : argmax + 1);
		hist_prev = hist[prev];
	//	hist_curr = hist[lid0];
		hist_next = hist[next];

//		if (maxval < 0.0f) {
//			hist_prev = -hist_prev;
//			maxval = -maxval;
//			hist_next = -hist_next;
//		}
		interp = 0.5f * (hist_prev - hist_next) / (hist_prev - 2.0f * maxval + hist_next);
		angle = (argmax + 0.5f + interp) / 18.0f;
//			2pi periodicity
		if (angle>=2)angle-=2;
		else if (angle<0)angle+=2;

		k.s0 = k.s2 *octsize; //c
		k.s1 = k.s1 *octsize; //r
		k.s2 = k.s3 *octsize; //sigma
		k.s3 = (angle - 1.0f) * M_PI_F;	   	  //angle
		keypoints[groupid] = k;
	}
//		An orientation is now assigned to our current keypoint.
//		We can create new keypoints of same (x,y,sigma) but a different angle.
//		For every local peak in histogram, every peak of value >= 80% of maxval generates a new keypoint
	barrier(CLK_GLOBAL_MEM_FENCE);
//	Everybody reads from contral memory
	k = keypoints[groupid];
	//	Now broadcast the result:
	argmax = pos[0];
	maxval = hist2[0];

	if ((lid0<36)&&(lid0 != argmax)){
		prev = (lid0 == 0 ? 35 : lid0 - 1);
		next = (lid0 == 35 ? 0 : lid0 + 1);
		hist_prev = hist[prev];
		hist_curr = hist[lid0];
		hist_next = hist[next];
		if ((hist_curr > hist_prev) && (hist_curr > hist_next) && (hist_curr >= 0.8f * maxval)) {
			/* Use parabolic fit to interpolate peak location from 3 samples. Set angle in range -PI to PI. */
//			if (hist_curr < 0.0f) {
//				hist_prev = -hist_prev;
//				hist_curr = -hist_curr;
//				hist_next = -hist_next;
//			}
			//if (hist_curr >= hist_prev  &&  hist_curr >= hist_next)
			interp = 0.5f * (hist_prev - hist_next) / (hist_prev - 2.0f * hist_curr + hist_next);

			angle = (lid0 + 0.5f + interp) / 18.0f;
//			2pi periodicity
			if (angle>=2)angle-=2;
			else if (angle<0) angle+=2;
			k.s3 = (angle-1.0f)*M_PI_F;
			old  = atomic_inc(counter);
			if (old < nb_keypoints)
				keypoints[old] = k;
		} //end "val >= 80%*maxval"
	} //end loop in histogram
}




__kernel void descriptor(
	__global keypoint* keypoints,
	__global unsigned char *descriptors,
	__local float* tmp_descriptors,
	__global float* grad,
	__global float* orim,
	int keypoints_start,
	int keypoints_end,
	int grad_width,
	int grad_height)
{

	int gid0 = (int) get_global_id(0);
	if (keypoints_start <= gid0 && gid0 < keypoints_end) {

		keypoint k = keypoints[gid0];
		if (k.s1 != -1.0f) {

		/* Add features to vec obtained from sampling the grad and ori images
		   for a particular scale.  Location of key is (scale,row,col) with respect
		   to images at this scale.  We examine each pixel within a circular
		   region containing the keypoint, and distribute the gradient for that
		   pixel into the appropriate bins of the index array.
		*/
			int i,j;
			/*
				Local memory memset
			*/
			for (i=0; i < 128; i++)
				tmp_descriptors[128*gid0+i] = 0.0f;

			float rx, cx;
			int	irow = (int) (k.s1 + 0.5f), icol = (int) (k.s0 + 0.5f);
			float sine = sin(k.s3), cosine = cos(k.s3);

			/* The spacing of index samples in terms of pixels at this scale. */
			float spacing = k.s2 * 3;

			/* Radius of index sample region must extend to diagonal corner of
			index patch plus half sample for interpolation. */
			int iradius = (int) ((1.414f * spacing * 2.5f) + 0.5f);

			/* Examine all points from the gradient image that could lie within the index square. */

			for (i = -iradius; i <= iradius; i++) {
				for (j = -iradius; j <= iradius; j++) {

					/* Makes a rotation of -angle to achieve invariance to rotation */
					 rx = ((cosine * i - sine * j) - (k.s1 - irow)) / spacing + 1.5f;
					 cx = ((sine * i + cosine * j) - (k.s0 - icol)) / spacing + 1.5f;

					 /* Compute location of sample in terms of real-valued index array
					 coordinates.  Subtract 0.5 so that rx of 1.0 means to put full
					 weight on index[1] (e.g., when rpos is 0 and 4 is 3. */

					/* Test whether this sample falls within boundary of index patch. */ //FIXME: cast to int for comparison
					if (rx > -1.0f && rx < 4.0f && cx > -1.0f && cx < 4.0f
						 && (irow +i) >= 0  && (irow +i) < grad_height && (icol+j) >= 0 && (icol+j) < grad_width) {

						/* Compute Gaussian weight for sample, as function of radial distance
				 		  from center.  Sigma is relative to half-width of index. */
						float mag = grad[(int)(icol+j) + (int)(irow+i)*grad_width]
									 * exp(- 0.125f*((rx - 1.5f) * (rx - 1.5f) + (cx - 1.5f) * (cx - 1.5f)) );

						/* Subtract keypoint orientation to give ori relative to keypoint. */
						float ori = orim[(int)(icol+j)+(int)(irow+i)*grad_width] -  k.s3;

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
						if (ri >= -1  &&  ri < 4  && oi >=  0  &&  oi <= 8  && rfrac >= 0.0f  &&  rfrac <= 1.0f) {

						/* Put appropriate fraction in each of 8 buckets around this point
							in the (row,col,ori) dimensions.  This loop is written for
							efficiency, as it is the inner loop of key sampling. */
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
													we want descriptors([rindex][cindex][oindex])[gid0]
														rindex in [0,3]
													 	cindex in [0,3]
													 	oindex in [0,7]
													so	rindex*4 + cindex is in [0,15]
														i = (rindex*4+cindex)*8 + oindex is in [0,127]
													finally : descriptors[128*gid0+i]
														with a vertical representation
												*/

												tmp_descriptors[128*gid0+(rindex*4 + cindex)*8+oindex]
													+= (cweight * ((orr == 0) ? 1.0f - ofrac : ofrac));



											} //end for
										} //end "valid cindex"
									}
								} //end "valid rindex"
							}
						}
					} //end "sample in boundaries"
				} //end "j loop"
			} //end "i loop"



			/*
				At this point, we have a descriptor associated with our keypoint.
				We have to normalize it, then cast to 1-byte integer
			*/


			// Normalization
			float norm = 0;
			for (i = 0; i < 128; i++)
				norm+=pow(tmp_descriptors[128*gid0+i],2); //warning: not the same as C "pow"
			norm = rsqrt(norm); //norm = 1.0f/sqrt(norm); //half_rsqrt to speed-up
			for (i = 0; i < 128; i++)
				tmp_descriptors[128*gid0+i] *= norm;


			//Threshold to 0.2 of the norm, for invariance to illumination
			bool changed = false;
			norm = 0;
			for (i = 0; i < 128; i++) {
				if (tmp_descriptors[128*gid0+i] > 0.2f) {
					tmp_descriptors[128*gid0+i] = 0.2f;
					changed = true;
				}
				norm += pow(tmp_descriptors[128*gid0+i],2);
			}

			//if values have been changed, we have to normalize again...
			if (changed) {
				norm = rsqrt(norm);
				for (i = 0; i < 128; i++)
					tmp_descriptors[128*gid0+i] *= norm;
			}

			//finally, cast to integer
			//store to global memory : tmp_descriptor[i][gid0] --> descriptors[i][gid0]
			for (i = 0; i < 128; i++) {
				descriptors[128*gid0+i]
					= (unsigned char) MIN(255,(unsigned char)(512.0f*tmp_descriptors[128*gid0+i]));
					//= (unsigned char) tmp_descriptors[128*gid0+i];
			}


		} //end "valid keypoint"
	} //end "in the keypoints"
}


