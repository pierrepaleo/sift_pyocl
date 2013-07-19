#define WORKGROUP_SIZE 64

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
	if (!(start <= gid && gid < end))
		return;
		
	float dist1 = 1000000000000.0f, dist2 = 1000000000000.0f;
	int current_min = 0;
	int old;
	
	__local unsigned char desc1[WORKGROUP_SIZE]; //store the descriptor of keypoint we are looking (in list 1)
	__local uint3 candidates[WORKGROUP_SIZE]

	desc1[lid0] = ((keypoints1[gid]).desc)[lid0];
	
	int frac = end/WORKGROUP_SIZE; //fraction of the list that will be processed by a thread
	int low_bound = lid0*frac;
	int up_bound = low_bound+frac;
	if (lid0 == WORKGROUP_SIZE -1) up_bound += end%WORKGROUP_SIZE; //fix the workgroup size to do a bitwise shift 

	
	for (int i = low_bound; i<up_bound; i++) {
	
		unsigned int dist = 0;
		for (int j=0; j<128; j++) { //parallel reduction on another workgroup dimension ?
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
	
	if (lid0 == low_bound) candidates[lid0/frac] = (uint3) (dist1, dist2, current_min)
	barrier(CLK_LOCAL_MEM_FENCE);
	
		//Now each block has its pair of best candidates (dist1,dist2) at position current_min
		//Find the minimum of them all, and the "second minimum" : (min1,min2)
		
	
	if (lid0 == 0) {
		unsigned int current_min0, dist10, dist_20, min1=4294967296, min2=4294967296, abs_min = 0;
		for (int i = 0; i < WORKGROUP_SIZE; i++) {
			
			(dist10, dist20, current_min0) = candidates[i]; //is it OK ?
			dist = dist10;
			if (dist < dist10) {
				min2 = dist10;
				min1 = dist;
				abs_min = i;
			} 
			else if (dist < dist2) {
				min2 = dist;
			}
			
			dist = dist20;
			if (dist < dist10) {
				min2 = dist10;
				min1 = dist;
				abs_min = i;
			} 
			else if (dist < dist2) {
				min2 = dist;
			}
		}//end for
	
		if ((min1/min2 < ratio_th && gid <= abs_min)) {
			uint2 pair = 0;
			pair.s0 = (unsigned int) gid;
			pair.s1 = (unsigned int) current_min;
			old = atomic_inc(counter);
			if (old < max_nb_keypoints) matchings[old] = pair;
		}
	}//end lid0 == 0

}


















