def calc_size(shape, blocksize):
    """
    Calculate the optimal size for a kernel according to the workgroup size
    """
    if "__len__" in dir(blocksize):
        return tuple((i + j - 1) & ~(j - 1) for i, j in zip(shape, blocksize))
    else:
        return tuple((i + blocksize - 1) & ~(blocksize - 1) for i in shape)
