#define MIN(i,j) ( (i)<(j) ? (i):(j) )


#define DOUBLEMIN(a,b,c,d) ((a) < (c) ? ((b) < (c) ? (int2)(a,b) : (int2)(a,c)) : ((a) < (d) ? (int2)(c,a) : (int2)(c,d)))

#define ABS4(q1,q2) (int) (((int) (q1.s0 < q2.s0 ? q2.s0-q1.s0 : q1.s0-q2.s0)) + ((int) (q1.s1 < q2.s1 ? q2.s1-q1.s1 : q1.s1-q2.s1))+ ((int) (q1.s2 < q2.s2 ? q2.s2-q1.s2 : q1.s2-q2.s2)) + ((int) (q1.s3 < q2.s3 ? q2.s3-q1.s3 : q1.s3-q2.s3)))


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






__kernel void matching(
	__global t_keypoint* keypoints1,
	__global t_keypoint* keypoints2,
	__global int2* matchings,
	__global int* counter,
	int max_nb_keypoints,
	float ratio_th,
	int size1,
	int size2)
{
	int gid0 = get_global_id(0);
	if (!(0 <= gid0 && gid0 < size1))
		return;

	float dist1 = 1000000000000.0f, dist2 = 1000000000000.0f; //HUGE_VALF ?
	int current_min = 0;
	int old;

	//pre-fetch
	unsigned char desc1[128];
	for (int i = 0; i<128; i++)
		desc1[i] = ((keypoints1[gid0]).desc)[i];

	//each thread gid0 makes a loop on the second list
	for (int i = 0; i<size2; i++) {

		//L1 distance between desc1[gid0] and desc2[i]
		int dist = 0;
		for (int j=0; j<128; j++) { //1 thread handles 4 values (uint4) = 
			unsigned char dval1 = desc1[j], dval2 = ((keypoints2[i]).desc)[j];
			dist += ((dval1 > dval2) ? (dval1 - dval2) : (-dval1 + dval2));

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

	if (dist2 != 0 && dist1/dist2 < ratio_th) {
		int2 pair = 0;
		pair.s0 = gid0;
		pair.s1 = current_min;
		old = atomic_inc(counter);
		if (old < max_nb_keypoints) matchings[old] = pair;
	}
}
