#!/usr/bin/env python
import numpy
def my_gradient(mat):
    """
    numpy implementation of gradient :
    "The gradient is computed using central differences in the interior and first differences at the boundaries. The returned gradient hence has the same shape as the input array."
    """
    g = numpy.gradient(mat)
    return numpy.sqrt(g[0]**2+g[1]**2), numpy.arctan2(g[0],g[1]) #image.cl/compute_gradient_orientation() puts a "-" here
    
    
def my_local_maxmin(dog_prev,dog,dog_next,thresh,border_dist,octsize,EdgeThresh0,EdgeThresh,nb_keypoints,s):
    """
    a python implementation of 3x3 maximum (positive values) or minimum (negative or null values) detection
    an extremum candidate "val" has to be greater than 0.8*thresh
    The three DoG have the same size.
    """
    output = -numpy.ones((nb_keypoints,4),dtype=numpy.float32) #for invalid keypoints
    width = dog.shape[1]
    height = dog.shape[0]
    counter = 0
    
    for j in range(border_dist,width - border_dist):
        for i in range(border_dist,height - border_dist):
            val = dog[i,j]
            if (numpy.abs(val) > 0.8*thresh): #keypoints refinement: eliminating low-contrast points
                if (is_maxmin(dog_prev,dog,dog_next,val,i,j,octsize,EdgeThresh0,EdgeThresh) != 0):
                	output[counter,0]=val
                	output[counter,1]=i
                	output[counter,2]=j
                	output[counter,3]=s
                	counter+=1   	      	
    return output
    
    
def is_maxmin(dog_prev,dog,dog_next,val,i0,j0,octsize,EdgeThresh0,EdgeThresh):
    """
    return 1 iff mat[i0,j0] is a local (3x3) maximum
    return -1 iff mat[i0,j0] is a local (3x3) minimum
    return 0 by default (neither maximum nor minimum, or value on an edge)
     * Assumes that we are not on the edges, i.e border_dist >= 2 above
    """
    ismax = 0
    ismin = 0
    res = 0
    if (val > 0.0): ismax = 1
    else: ismin = 1
    for j in range(j0-1,j0+1+1):
        for i in range(i0-1,i0+1+1):
            if (ismax == 1):
                if (dog_prev[i,j] > val or dog[i,j] > val or dog_next[i,j] > val): ismax = 0
            if (ismin == 1):
                if (dog_prev[i,j] < val or dog[i,j] < val or dog_next[i,j] < val): ismin = 0;
    
    if (ismax == 1): res =  1 
    if (ismin == 1): res = -1
    
    #keypoint refinement: eliminating points at edges
    H00 = dog[i0-1,j0] - 2.0 * dog[i0,j0] + dog[i0+1,j0]
    H11 = dog[i0,j0-1]- 2.0 * dog[i0,j0] + dog[i0,j0+1]
    H01 = ( (dog[i0+1,j0+1] - dog[i0+1,j0-1])
		- (dog[i0-1,j0+1] - dog[i0-1,j0-1]) ) / 4.0;

    det = H00 * H11 - H01 * H01
    trace = H00 + H11

    if (octsize <= 1):
        thr = EdgeThresh0
    else:
        thr = EdgeThresh
    if (det < thr * trace * trace):
        res = 0
        
    return res
    




def my_interp_keypoint(dog_prev,dog,dog_next, s, r, c,movesRemain,peakthresh,width,height):
    ''''
     A Python implementation of SIFT "InterpKeyPoints"
     (s,r,c) : coords of the processed keypoint in the scale space
     WARNING: replace "1.6" by "InitSigma" if InitSigma has not its default value 
     The recursive calls were replaced by a loop.
    '''
    
    newr = r
    newc = c
    loop = 1
    movesRemain = 5
    while (loop == 1):
    
        x,peakval = fit_quadratic(dog_prev,dog,dog_next, newr, newc)
        
        
        if (x[1] > 0.6 and newr < height - 3):
            newr+=1
        elif (x[1] < -0.6 and newr > 3):
            newr-=1
        if (x[2] > 0.6 and newc < width - 3):
            newc+=1
        elif (x[2] < -0.6 and newc > 3):
            newc-=1
 
        #loop test
        if (movesRemain > 0  and  (newr != r or newc != c)):
            movesRemain-=1
        else:
            loop = 0
            

    if (abs(x[0]) <  1.5 and abs(x[1]) <  1.5 and abs(x[2]) <  1.5 and abs(peakval) > peakthresh):
        ki = numpy.zeros(4,dtype=numpy.float32)
        ki[0] = peakval
        ki[1] = r + x[1]
        ki[2] = c + x[2]
        ki[3] = 1.6 * 2.0**((float(s) + x[0]) / 3.0) #3.0 is "par.Scales" 
    else:
        ki = (-1,-1,-1,-1)
    
    return ki #our interpolated keypoint





def fit_quadratic(dog_prev,dog,dog_next, r, c):
    '''
    quadratic interpolation arround the keypoint (s,r,c)
    '''

    #gradient
    g = numpy.zeros(3,dtype=numpy.float32)
    g[0] = (dog_next[r,c] - dog_prev[r,c]) / 2.0
    g[1] = (dog[r+1,c] - dog[r-1,c]) / 2.0;
    g[2] = (dog[r,c+1] - dog[r,c-1]) / 2.0
	#hessian
    H = numpy.zeros((3,3)).astype(numpy.float32)
    H[0][0] = dog_prev[r,c]   - 2.0 * dog[r,c] + dog_next[r,c]
    H[1][1] = dog[r-1,c] - 2.0 * dog[r,c] + dog[r+1,c]
    H[2][2] = dog[r,c-1] - 2.0 * dog[r,c] + dog[r,c+1]
    H[0][1] = H[1][0] = ( (dog_next[r+1,c] - dog_next[r-1,c])
    		 			- (dog_prev[r+1,c] - dog_prev[r-1,c]) ) / 4.0


    H[0][2] = H[2][0] = ( (dog_next[r,c+1] - dog_next[r,c-1])
		    		 - (dog_prev[r,c+1] - dog_prev[r,c-1]) ) / 4.0

    H[1][2] = H[2][1]= ( (dog[r+1,c+1] - dog[r+1,c-1])
    				 - (dog[r-1,c+1] - dog[r-1,c-1]) ) / 4.0
    		 
    x = -numpy.dot(numpy.linalg.inv(H),g) #extremum position
    peakval = dog[r,c] + 0.5 * (x[0]*g[0]+x[1]*g[1]+x[2]*g[2])
	
    return x, peakval

