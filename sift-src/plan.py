
"""
Contains a class for creating a plan, allocating arrays, compiling kernels and other things like that
"""
import gc
import numpy
import pyopencl, pyopencl.array
from .param import par
from .opencl import ocl

class SiftPlan(object):
    """
    How to calculate a set of SIFT keypoint on an image:
    
    siftp = sift.SiftPlan(img.shape,img.dtype,devicetype="GPU")
    kp = siftp.keypoints(img)
    
    kp is a nx132 array. the second dimension is composed of x,y, scale and angle as well as 128 floats describing the keypoint  
     
    """
    def __init__(self, shape=None, dtype=None, devicetype="GPU", template=None, profile=False):
        """
        Contructor of the class
        """
        if template is not None:
            self.shape = template.shape
            self.dtype = template.dtype
        else:
            self.shape = shape
            self.dtype = dtype
        self.profile = bool(profile)
        self.scales = []
        self.buffers = {}
        self.memory = None
        self._calc_scales()
        self._calc_memory()
        self.device = ocl.select_device(type=devicetype, memory=self.memory, best=True)
        self.ctx = ctx = pyopencl.Context(devices=[pyopencl.get_platforms()[self.device[0]].get_devices()[self.device[1]]])
        if profile:
            self.queue = pyopencl.CommandQueue(self.ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
        else:
            self.queue = pyopencl.CommandQueue(self.ctx)
        self._allocate_buffers()

    def __del__(self):
        """
        Destructor: release all buffers
        """
        self._free_kernels()
        self._free_buffers()
        self.queue = None
        self.ctx = None
        gc.collect()

    def _calc_scales(self):
        self.scales = [self.shape]
        shape = self.shape
        min_size = 2 * par.BorderDist + 2
        while min(shape) > min_size:
            shape = [i // 2 for i in shape]
            self.scales.append(shape)
        self.scales.pop()

    def _calc_memory(self):
        # Just the context + kernel takes about 75MB on the GPU
        self.memory = 75 * 2 ** 20
        size_of_float = numpy.dtype(numpy.float32).itemsize
        size_of_input = numpy.dtype(self.dtype).itemsize
        #raw images:
        size = self.shape[0] * self.shape[1]
        self.memory += size * (size_of_float + size_of_input) #raw_float + initial_image
        for scale in self.scales:
            nr_blur = par.Scales + 3
            nr_dogs = par.Scales + 2
            size = scale[0] * scale[1]
            self.memory += size * (nr_blur + nr_dogs) * size_of_float

    def _allocate_buffers(self):
        self.buffers["raw"] = pyopencl.array.empty(self.queue, self.shape, dtype=self.dtype, order="C")
        self.buffers["input"] = pyopencl.array.empty(self.queue, self.shape, dtype=numpy.float32, order="C")
        for octave, scale in enumerate(self.scales):

            for i in range(par.Scales + 3):
                self.buffers[(octave, i, "G") ] = pyopencl.array.empty(self.queue, scale, dtype=numpy.float32, order="C")
            for i in range(par.Scales + 2):
                self.buffers[(octave, i, "DoG") ] = pyopencl.array.empty(self.queue, scale, dtype=numpy.float32, order="C")
    def _free_buffers(self):
        """
        free all memory allocated on the device
        """
        for buffer_name in self.buffer:
            if self.buffer[buffer_name] is not None:
                try:
                    self.buffer[buffer_name].release()
                    self.buffer[buffer_name] = None
                except pyopencl.LogicError:
                    logger.error("Error while freeing buffer %s" % buffer_name)

    def _compile_kernels(self, kernel_file=None):
        """
        Call the OpenCL compiler
        @param kernel_file: path tothe
        """
        return
#        kernel_name = "ocl_azim_LUT.cl"
#        if kernel_file is None:
#            if os.path.isfile(kernel_name):
#                kernel_file = os.path.abspath(kernel_name)
#            else:
#                kernel_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), kernel_name)
#        else:
#            kernel_file = str(kernel_file)
#        kernel_src = open(kernel_file).read()
#
#        compile_options = "-D NBINS=%i  -D NIMAGE=%i -D NLUT=%i -D ON_CPU=%i" % \
#                (self.bins, self.size, self.lut_size, int(self.device_type == "CPU"))
#        logger.info("Compiling file %s with options %s" % (kernel_file, compile_options))
#        try:
#            self._program = pyopencl.Program(self._ctx, kernel_src).build(options=compile_options)
#        except pyopencl.MemoryError as error:
#            raise MemoryError(error)

    def _free_kernels(self):
        """
        free all kernels
        """
#        for kernel in self._cl_kernel_args:
#            self._cl_kernel_args[kernel] = []
        self._program = None
