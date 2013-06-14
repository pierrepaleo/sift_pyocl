#define ORDERV(x,a,b) { bool swap = reverse ^ (getKey(x[a])<getKey(x[b])); \
      float auxa = x[a]; float auxb = x[b]; \
      x[a] = (swap)?auxb:auxa; x[b] = (swap)?auxa:auxb; }
#define B2V(x,a) { ORDERV(x,a,a+1) }
#define B4V(x,a) { for (int i4=0;i4<2;i4++) { ORDERV(x,a+i4,a+i4+2) } B2V(x,a) B2V(x,a+2) }
#define B8V(x,a) { for (int i8=0;i8<4;i8++) { ORDERV(x,a+i8,a+i8+4) } B4V(x,a) B4V(x,a+4) }
#define B16V(x,a) { for (int i16=0;i16<8;i16++) { ORDERV(x,a+i16,a+i16+8) } B8V(x,a) B8V(x,a+8) }
#define getKey(a) (a)
#define getValue(a) (0)
#define makeData(k,v) (k)

/*
 * 
 * Sort in place !!!
 * 1D kernel
 * 
 * data: guess !
 * inc0: number of times we need to pass to get a full sort
 * dir: direction ?
 * aux: local shared memory
 */


__kernel void sort(
		__global float * data,
		int inc0,
		int dir,
		__local float * aux,
		int size)
{
  int t = get_global_id(0); // thread index
  int wgBits = 4*get_local_size(0) - 1; // bit mask to get index in local memory AUX (size is 4*WG)
  int inc,low,i;
  bool reverse;
  float x[4];

  if (t<size){
	  // First iteration, global input, local output
	  inc = inc0>>1;
	  low = t & (inc - 1); // low order bits (below INC)
	  i = ((t - low) << 2) + low; // insert 00 at position INC
	  reverse = ((dir & i) == 0); // asc/desc order
	  for (int k=0;k<4;k++) 
		  x[k] = data[i+k*inc];
	  B4V(x,0);
	  for (int k=0;k<4;k++) 
		  aux[(i+k*inc) & wgBits] = x[k];
	  barrier(CLK_LOCAL_MEM_FENCE);

	  // Internal iterations, local input and output
	  for ( ;inc>1;inc>>=2)
	  {
	    low = t & (inc - 1); // low order bits (below INC)
	    i = ((t - low) << 2) + low; // insert 00 at position INC
	    reverse = ((dir & i) == 0); // asc/desc order
	    for (int k=0;k<4;k++) 
	    	x[k] = aux[(i+k*inc) & wgBits];
	    B4V(x,0);
	    barrier(CLK_LOCAL_MEM_FENCE);
	    for (int k=0;k<4;k++) 
	    	aux[(i+k*inc) & wgBits] = x[k];
	    barrier(CLK_LOCAL_MEM_FENCE);
	  }

	  // Final iteration, local input, global output, INC=1
	  i = t << 2;
	  reverse = ((dir & i) == 0); // asc/desc order
	  for (int k=0;k<4;k++) 
		  x[k] = aux[(i+k) & wgBits];
	  B4V(x,0);
	  for (int k=0;k<4;k++) 
		  data[i+k] = x[k];
	  
  }
  
}
