/**
 * \brief Linear combination of two matrices
 *
 * @param u: Pointer to global memory with the input data of the first matrix
 * @param a: float scalar which multiplies the first matrix
 * @param v: Pointer to global memory with the input data of the second matrix
 * @param b: float scalar which multiplies the second matrix
 * @param w: Pointer to global memory with the output data
 * @param width: integer, number of columns the matrices
 * @param height: integer, number of lines of the matrices
 *
 */
 
__kernel void combine(
	__global float *u,
	float a,
	__global float *v,
	float b,
	__global float *w,
	int dog,
	int width,
	int height)
{
	
	int gid1 = (int) get_global_id(1);
	int gid0 = (int) get_global_id(0);

	if (gid0 < height && gid1 < width) {
	
		int index = gid0 * width + gid1;
		int index_dog = dog * width * height +  index;
		w[index_dog] = a * u[index] + b * v[index];
	}
}
