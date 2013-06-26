/********************************************************************
 * A small code to compare keypoints of sift.cpp and our opencl SIFT
 *
 * Usage : make && ./compare extrema_cpp.txt extrema_opencl.txt
 *
 * Format of the keypoints in the files :
 [  -3.41704249  368.90612793   14.99513245    2.78045988]
 [  -6.37356377  449.18521118   69.4825592     2.53365469]
  
 ********************************************************************
*/

#include "compare.h"
#define MAX_KP 300 //for a given octave 
#define DIGITS 3 //for comparison precision (10e-DIGITS)


/*
  Swap two keypoints pointers
*/
void keypoint_swap(keypoint* k1, keypoint* k2) {
	keypoint tmp_ptr = *k1;
	*k1 = *k2;
	*k2 = tmp_ptr; 
}


/*
truncates "a" to its "digits" first digits
example: truncate(3.141592,3) = 3.141000

warning: have to use round(), for compiler removes the cast to (int)
WARNING: not 100% reliable, due to pre-truncating
*/

float truncate(float a,unsigned int digits) {
	if (digits != 0) {
		float p = pow(10,digits);
		float tmp = (int) round(a*p);
		return ((int) tmp)/p;
	}
	else return (float) ((int) a);
}




int parse_keypoints(char* filename, keypoint* keypoints, unsigned int* total_keypoints) {
	FILE* stream = fopen(filename,"r+");
	if (!stream) { printf("Error: Could not open file %s\n",filename); return -1; }
	float p=0,r=0,c=0,s=0;
	keypoint* kp = &(keypoints[0]);
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
					kp = (keypoint*) &keypoints[k+1]; 
					k++;
					break;
			} //end switch
			j++;
		} //end isdigit
	} //end read loop
	//puts a "end of keypoints" marker
	keypoints[k].r = -1.0;
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
	
	unsigned int i,j, kp_ok = 0;
	float r=0,c=0,s=0;
	int flag_ok = 0;

	for (i=0; total_keypoints_opencl > i; i++) {
		if (k_opencl[i].r != -1.0) {
			r = truncate(k_opencl[i].r,DIGITS);
			c = truncate(k_opencl[i].c,DIGITS);
			s = truncate(k_opencl[i].s,DIGITS);
			flag_ok = 0;
			for (j=0; flag_ok == 0 && total_keypoints_cpp > j; j++) {
				if (k_cpp[j].r != -1.0) {
					if (truncate(k_cpp[j].r,DIGITS) == r 
						&& truncate(k_cpp[j].c,DIGITS) == c
						/*&& truncate(k_cpp[j].s,DIGITS) == s CHECK THAT LATER*/) {
						flag_ok = 1;
						kp_ok++;
					}
				}
			}
			if (flag_ok == 0) printf("Keypoint (%f,%f,%f) did not match\n",r,c,s);
		}
	}
	printf("End of comparison -- %d/(%d,%d) keypoints matches\n",
		kp_ok,total_keypoints_opencl,total_keypoints_cpp);
	
	keypoint* output = (keypoint*) calloc(total_keypoints_opencl-1,sizeof(keypoint));
//	puts("before cut sort");
	for(i=0;6 >= i; i++) printf("%f ",k_opencl[i].r);
	//puts(""); puts("after cut sort");
	merge_sort(k_opencl,0,6/*total_keypoints_opencl-1*/,output);
	//for(i=0;30 > i; i++) printf("%f ",k_opencl[i].r);
	puts("");
	free(k_cpp);
	free(k_opencl);
	free(output);
	return 1;


}

//[start,...,end]
void merge_sort(keypoint* input, unsigned int start, unsigned int end, keypoint* output) {
	unsigned int len = end-start+1;
	unsigned int middle = (end+start)/2;
	if (len > 2) {
		merge_sort(input, start, middle,output);
		merge_sort(input,middle+1,end,output);
	}
	else {
		if (len == 2) {
			if (input[start].r > input[end].r) keypoint_swap(&input[start],&input[end]);
		}
	}
/*	printf("Call to %d %d %d\n",start,middle,end);*/
	if (len > 2) merge(input, output, start, middle, end);
}
/*
	Merge 2 sorted lists
*/
void merge(keypoint* input, keypoint* output, unsigned int start, unsigned int middle, unsigned int end) {
	int i1 = 0, i2 = 0, stop = 0;
	printf("I am %d %d %d\n",start,middle,end);
	while (!stop) {
		if (input[start+i1].r > input[middle+1+i2].r) {
			output[start+i1+i2] = input[middle+1+i2];
			i2++;
		}
		else {
			output[start+i1+i2] = input[start+i1];
			i1++;
		}
		if (i1 == middle+1) stop = 1;
		if (i2 == end+1) stop = 2;
	}
	int i;
	if (stop == 1) //recopy the end of the 2nd list 
		for (i = i2; end >= i; i++) output[start+i1+i] = input[middle+1+i];
	if (stop == 2) //recopy the end of the 1st list
		for (i = i1; middle >= i; i++) output[start+i2+i] = input[start+i];
	

	puts("Merged :");
	for (i = 0; i1+i2 >= i; i++) printf("%f ",output[start+i].r);
	puts("");
	
}
























