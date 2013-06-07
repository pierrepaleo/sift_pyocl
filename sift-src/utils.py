from math import ceil

def calc_size(shape, blocksize):
    """
    Calculate the optimal size for a kernel according to the workgroup size
    """
    if "__len__" in dir(blocksize):
        return tuple((i + j - 1) & ~(j - 1) for i, j in zip(shape, blocksize))
    else:
        return tuple((i + blocksize - 1) & ~(blocksize - 1) for i in shape)


def kernel_size(sigma, odd=False, cutoff=4):
    """
    Calculate the optimal kernel size for a convolution with sigma
    
    @param sigma: width of the gaussian 
    @param odd: enforce the kernel to be odd (more precise ?)
    """
    size = int(ceil(2 * cutoff * sigma + 1))
    if odd and size % 2 == 0:
        size += 1
    return size

