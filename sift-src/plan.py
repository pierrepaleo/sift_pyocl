
"""
Contains a class for creating a plan, allocating arrays, compiling kernels and other things like that
"""
import gc
import numpy
import pyopencl, pyopencl.array
from .param import par
from .opencl import ocl
from .utils import calc_size

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
        self.programs = {}
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
        self._compile_kernels()

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
            self.memory += size * (nr_blur + nr_dogs + 1) * size_of_float  # 1 temporary array

    def _allocate_buffers(self):
        self.buffers["raw"] = pyopencl.array.empty(self.queue, self.shape, dtype=self.dtype, order="C")
        self.buffers["input"] = pyopencl.array.empty(self.queue, self.shape, dtype=numpy.float32, order="C")
        for octave, scale in enumerate(self.scales):
            self.buffers[(octave, "tmp") ] = pyopencl.array.empty(self.queue, scale, dtype=numpy.float32, order="C")
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

    def _compile_kernels(self):
        """
        Call the OpenCL compiler
        """
        kernels = ["convolution", "preprocess"]
        for kernel in kernels:
            kernel_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), kernel + ".cl")
            kernel_src = open(kernel_file).read()
            try:
                program = pyopencl.Program(self.ctx, kernel_src).build()
            except pyopencl.MemoryError as error:
                raise MemoryError(error)
            self.programs[kernel] = program

    def _free_kernels(self):
        """
        free all kernels
        """
        self.programs = {}

    def calc_workgroups(self):
        """
        First try to guess the best workgroup size, then calculate all global worksize
        """
        device = self.ctx.devices[0]
        max_work_group_size = device.max_work_group_size
        max_work_item_sizes = device.max_work_item_sizes
