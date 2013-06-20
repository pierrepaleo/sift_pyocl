/********************************************************************
 * A small code to compare keypoints of sift.cpp and our opencl SIFT
 *
 * Usage : make && ./compare extrema_cpp.txt extrema_opencl.txt
 *
 ********************************************************************
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define MAX_KP 300 //for a given octave 
#define DIGITS 3 //for comparison precision (10e-DIGITS)

typedef struct keypoint {
	float p;
	float r;
	float c;
	float s;
} keypoint;

/*
  Swap two keypoints pointers
*/
void keypoint_swap(keypoint k1, keypoint k2) {
	keypoint tmp_ptr = k1;
	k1 = k2;
	k2 = tmp_ptr; 
}




/*
truncates "a" to its "digits" first digits
example: truncate(3.141592,3) = 3.141000
*/

float truncate(float a,unsigned int digits) {
	if (digits != 0) {
		float p = pow(10,digits);
		return (float) ((int) (a*p))/p;
	}
	else return (float) ((int) a);
}






int parse_keypoints(char* filename, keypoint* keypoints, unsigned int* total_keypoints) {
	FILE* stream = fopen(filename,"r+");
	if (!stream) { printf("Error: Could not open file %s\n",filename); return -1; }
	float p=0,r=0,c=0,s=0;
	keypoint* kp = (keypoint*) calloc(1,sizeof(keypoint));
	unsigned int j= 1, k =0;
	char str[511];
	while (EOF != fscanf(stream,"%s",str)) {
		if (strlen(str) > 3 || isdigit(str[0])) {
			switch (j & 3) {
				case 1:
					kp->p = atof(str);
					break;
				case 2:
					kp->r = atof(str);
					break;
				case 3:
					kp->c = atof(str);
					break;
				case 0:
					kp->s = atof(str);
					keypoints[k] = *kp;
					kp = (keypoint*) calloc(1,sizeof(keypoint));
					k++;
					break;
			} //end switch
			j++;
		} //end isdigit
	} //end read loop
	//puts a "end of keypoints" marker
	kp = (keypoint*) calloc(1,sizeof(keypoint));
	kp->r = -1.0;
	keypoints[k] = *kp;
	*total_keypoints = k;
	return 1;
}






int main(int args, char* argv[]) {
	if (3 > args) { printf("Usage: %s filename_opencl.txt filename_cpp.txt\n",argv[0]); return -1; }

	unsigned int total_keypoints_cpp = 0, total_keypoints_opencl = 0;
	keypoint* k_cpp = (keypoint*) calloc(MAX_KP,sizeof(keypoint));
	keypoint* k_opencl = (keypoint*) calloc(MAX_KP,sizeof(keypoint));
	if (parse_keypoints(argv[2],k_cpp,&total_keypoints_cpp) == -1) return -1;
	if (parse_keypoints(argv[1],k_opencl,&total_keypoints_opencl) == -1) return -1;
	//let the fun begin
	
	unsigned int i,j, kp_ok = 0;
	float r=0,c=0,s=0;
	int flag_ok = 0;

	for (i=0; MAX_KP > i && k_opencl[i].r != -1.0; i++) {
		r = truncate(k_opencl[i].r,DIGITS);
		c = truncate(k_opencl[i].c,DIGITS);
		s = truncate(k_opencl[i].s,DIGITS);
		flag_ok = 0;
		for (j=0; flag_ok == 0 && MAX_KP > j && k_cpp[j].r != -1.0; j++) {
			if (truncate(k_cpp[j].r,DIGITS) == r && truncate(k_cpp[j].c,DIGITS) == c && truncate(k_cpp[j].s,DIGITS) == s) {
				flag_ok = 1;
				kp_ok++;
			}
		}
	}
	printf("End of comparison -- %d/(%d,%d) keypoints matches (opencl,cpp)\n",
		kp_ok,total_keypoints_opencl,total_keypoints_cpp);
	return 1;


}


void cut_sort(keypoint* input, unsigned int start, unsigned int end) {
	unsigned int len = end-start+1;
	if (len > 2) {
		cut_sort(input, start, (end+start)/2);
		cut_sort(input,(end+start)/2+1,end);
	}
	else {
		if (len == 1) ; //return input[start];
		else {
			if (input[start].r > input[end].r) keypoint_swap(input[start],input[end]);
			//else ;
		}
	}	
	

}


//TODO: print kp_opencl, see if kp_opencl[:].r have been sorted by parts of 2






















