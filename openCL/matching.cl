#define WORKGROUP_SIZE 8
#define MIN(i,j) ( (i)<(j) ? (i):(j) )
#define MAX(i,j) ( (i)<(j) ? (j):(i) )

/*
	Keypoint (c, r, s, angle) without its descriptor
*/
typedef float4 keypoint;


/*
	Keypoint with its descriptor
*/
typedef struct t_keypoint {
	keypoint kp;
	unsigned char desc[128];
} t_keypoint;



/*
 *
 * \brief Compute SIFT descriptors matching for two lists of descriptors.
 *		
 * Use L1 distance for speed and atomic_add.
 *
 * @param desc1 Pointer to global memory with the first list of descriptors
 * @param desc2: Pointer to global memory with the second list of descriptors
 * @param matchings: Pointer to global memory with the output pair of matchings (keypoint1, keypoint2) 
 * @param counter:
 * @param keypoints_start : index start for processing
 * @param keypoints_end: end index for processing
 
 par.MatchRatio = 0.73;
par.MatchXradius = 1000000.0f;
par.MatchYradius = 1000000.0f;
	
	
*/	
	
/*
	optimizations:
		-if needed, pre-fetch descriptors in shared memory
		-each descriptor is processed by one group of thread, each looking in a different region of the vector
		-we have a N*128 vector: one thread per descriptor value ? (to avoid loops)
*/



__kernel void matching(
	__global t_keypoint* keypoints1,
	__global t_keypoint* keypoints2,
	__global uint2* matchings,
	__global int* counter,
	int max_nb_keypoints,
	float ratio_th,
	int start,
	int end)
{

	int gid0 = get_global_id(0);
	if (!(start <= gid0 && gid0 < end))
		return;
		
	float dist1 = 1000000000000.0f, dist2 = 1000000000000.0f; //HUGE_VALF ?
	int current_min = 0;
	int old;
	
	//pre-fetch
	__local unsigned char desc1[128];
	for (int i = 0; i<128; i++)
		desc1[i] = ((keypoints1[gid0]).desc)[i];
	
	//each thread gid0 makes a loop on the second list
	for (int i = start; i<end; i++) {
	
		//L1 distance between desc1[gid0] and desc2[i]
		unsigned int dist = 0;
		for (int j=0; j<128; j++) {
			unsigned char dval1 = desc1[j], dval2 = ((keypoints2[i]).desc)[j];
			dist += ((dval1 > dval2) ? (dval1 - dval2) : (-dval1 + dval2)); //fabs() ?
		}
		
		if (dist < dist1) { //candidate better than the first
			dist2 = dist1;
			dist1 = dist;
			current_min = i;
		} 
		else if (dist < dist2) { //candidate better than the second (but not the first)
			dist2 = dist;
		}
		
	}//end "i loop"	
	

		//to avoid duplicata : gid0 <= current_min
	if ((dist1/dist2 < ratio_th && gid0 <= current_min)) {
	
		//pair of keypoints
		uint2 pair = 0;
		pair.s0 = (unsigned int) gid0;
		pair.s1 = (unsigned int) current_min;
		old = atomic_inc(counter);
		if (old < max_nb_keypoints) matchings[old] = pair;
	}

	
}





/*

	Let L2 be the length of "keypoints2" and W be the workgroup size.
	Each thread of the workgroup handles L2/W keypoints : [lid0*L2/W, (lid0+1)*L2/W[ ,
	 and gives a pair of "best distance / second-best distance" (d1,d2)
	Then, we take d1 = min{(d1,d2) | all threads} and d2 = second_min {(d1,d2) | all threads}

	 -----------------------------------------------
	|  thread 0 | thread 1 | ... | thread (W-1)    |
	 -----------------------------------------------
	 <---------->
	L2/W keypoints



	WORKGROUP SIZE IS FIXED TO 64

*/

__kernel void matching_v2(
	__global t_keypoint* keypoints1,
	__global t_keypoint* keypoints2,
	__global uint2* matchings,
	__global int* counter,
	int max_nb_keypoints,
	float ratio_th,
	int end)
{

	int gid = get_group_id(0);
	int lid0 = get_local_id(0); //[0,128[ !
	if (!(0 <= gid && gid < end))
		return;
		
	float dist1 = 1000000000000.0f, dist2 = 1000000000000.0f;
	int current_min = 0;
	int old;
	
	__local unsigned char desc1[64]; //store the descriptor of keypoint we are looking (in list 1)
	__local uint3 candidates[64];

	for (int i = 0; i < 2; i++)
		desc1[i*64+lid0] = ((keypoints1[gid]).desc)[i*64+lid0];

	int frac = (end >> 6)+1; //fraction of the list that will be processed by a thread
	int low_bound = lid0*frac;
	int up_bound = MIN(low_bound+frac,end);
	
	for (int i = low_bound; i<up_bound; i++) {
	
		unsigned int dist = 0;
		for (int j=0; j<128; j++) {
			unsigned char dval1 = desc1[j], dval2 = ((keypoints2[i]).desc)[j];
			dist += ((dval1 > dval2) ? (dval1 - dval2) : (-dval1 + dval2));
		}
		
		if (dist < dist1) {
			dist2 = dist1;
			dist1 = dist;
			current_min = i;
		} 
		else if (dist < dist2) {
			dist2 = dist;
		}
		
	}//end "i loop"	
	
	candidates[lid0] = (uint3) (dist1, dist2, current_min);
	barrier(CLK_LOCAL_MEM_FENCE);
	
		//Now each block has its pair of best candidates (dist1,dist2) at position current_min
		//Find the global minimum and the "second minimum" : (min1,min2)
		
	
	if (lid0 == 5) {
		unsigned int dist1_t, dist2_t, current_min_t, //values got from other threads
		dist0 = 0, dist10=429496729, dist20=429496729, index_abs_min = 0;
		
		for (int i = 0; i < 64; i++) {
			
			dist1_t = candidates[i].s0;
			dist2_t = candidates[i].s1;
			current_min_t = candidates[i].s2;

			//check if d1 can be a global min (or second_min)
			dist0 = dist1_t;
			if (dist0 < dist10) {
				dist20 = dist10;
				dist10 = dist0;
				index_abs_min = current_min_t;
			} 
			else if (dist0 < dist20) {
				dist20 = dist0;
			}
			//do the same with d2
			dist0 = dist2_t;
			if (dist0 < dist10) {
				dist20 = dist10;
				dist10 = dist0;
				index_abs_min = current_min_t;
			} 
			else if (dist0 < dist20) {
				dist20 = dist0;
			}
			
		}//end for
	
		if ((dist20 != 0 && (((float) dist10)/ ((float) dist20)) < ratio_th && gid <= index_abs_min)) {
			uint2 pair = 0;
			pair.s0 = (unsigned int) gid; //dist10; //gid; //current_min; //gid;
			pair.s1 = (unsigned int) index_abs_min; //candidates[61].s2-low_bound; //candidates[1].s2; //index_abs_min;
			old = atomic_inc(counter);
			if (old < max_nb_keypoints) matchings[old] = pair;
		}
	}//end lid0 == 0

}


















