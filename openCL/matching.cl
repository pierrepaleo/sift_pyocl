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
 *
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
	//__global t_keypoint* keypoints1,
	//__global t_keypoint* keypoints2,
	__global char* desc1, //FIXME: this is used for now
	__global char* desc2,
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
//	unsigned char desc1 = keypoints1.desc, desc2 = keypoints2.desc; //FIXME: uncomment
	
	//each thread gid0 makes a loop on the second list
	for (int i = start; i<end; i++) {
	
		//L1 distance between desc1[gid0] and desc2[i]
		unsigned int dist = 0;
		for (int j=0; j<128; j++) {
			unsigned char dval1 = desc1[gid0*128+j], dval2 = desc2[i*128+j];
			dist += ((dval1 > dval2) ? (dval1 - dval2) : (-dval1 + dval2)); //fabs() ?
			//TODO (?)  if (ABS(k1.x - k2.x) > par.MatchXradius || ABS(k1.y - k2.y) > par.MatchYradius) return tdist;
			//by default : par.MatchXradius = 1000000.0f
		}
		
		if (dist < dist1) { //candidate better than the first
			dist2 = dist1;
			dist1 = dist;
			current_min = i;
		} 
		else if (dist < dist2) { //candidate better than the second (but not the first)
			dist2 = dist;
		}
		
		if (dist1/dist2 < ratio_th) { //0.73^2. TODO: set this threshold as a parameter ?
		
			//pair of keypoints
			uint2 pair = 0;
			pair.s0 = (unsigned int) gid0;
			pair.s1 = (unsigned int) current_min;
			old = atomic_inc(counter);
			if (old < max_nb_keypoints) matchings[old] = pair;
		}
		
		
	}//end "i loop"	
	
}
