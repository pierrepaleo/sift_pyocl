__kernel void cholesky_decomposition(
	__global float* a, 
	__global float* p,
	int n)
{

	int gid0 = get_global_id(0);
	int gid1 = get_global_id(1);
	if (!(gid1 < n && gid1 <= gid0 && gid0 < n)) //for 0<=i<n ; i<=j<n
		return;

	int k;
	float sum = a[gid1*n+gid0];
	for (k=gid1-2;k>=0;k--) sum -= a[gid1*n+k]*a[gid0*n+k];
	if (gid1 == gid0) {
		if (sum <= 0.0) exit("error : not positive definite");
		p[gid1]=sqrt(sum); //no conflict since gid1 == gid0 ? otherwise, 1D workgroup then loop on rows
	} 
	else a[gid0*n+gid1]=sum/p[gid1]; //fix this division
}




/*

import scipy, scipy.misc, numpy, scipy.ndimage, pylab, scipy.optimize
import sift
angle = numpy.pi/5.0
#matrix = [[0.8,0.2],[-0.1,0.9]]
matrix = [[numpy.cos(angle),numpy.sin(angle)],[-numpy.sin(angle),numpy.cos(angle)]]
print("Matrix : %s" %matrix)

image1 = scipy.misc.lena()
image2 = scipy.ndimage.interpolation.affine_transform(image1,matrix,offset=numpy.array([0.0,0.0]),order=1, mode="constant")
s = sift.SiftPlan(template=image1,devicetype="gpu")
kp1 = s.keypoints(image1)
kp2 = s.keypoints(image2)
m = sift.MatchPlan(devicetype="GPU")
matching = m.match(kp1,kp2)
print matching.shape


N = matching.shape[0]
X = numpy.zeros((2*N,6))
X[::2,2:] = 1,0,0,0
X[::2,0] = matching.x[:,0]
X[::2,1] = matching.y[:,0]
X[1::2,0:3] = 0,0,0
X[1::2,3] = matching.x[:,0]
X[1::2,4] = matching.y[:,0]
X[1::2,5] = 1

y = numpy.zeros((2*N,1))
y[::2,0] = matching.x[:,1]
y[1::2,0] = matching.y[:,1]

#A = numpy.dot(X.transpose(),X)
#sol = numpy.dot(numpy.linalg.inv(A),numpy.dot(X.transpose(),y))

sol = numpy.dot(numpy.linalg.pinv(X),y)




/*
