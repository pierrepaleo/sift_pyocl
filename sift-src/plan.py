
"""
Contains a class for creating a plan, allocating arrays, compiling kernels and other things like that
"""
import time
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
            self.dtype = numpy.dtype(dtype)
        if len(self.shape) == 3:
            self.RGB = True
            self.shape = self.shape[:2]
        elif len(self.shape) == 2:
            self.RGB = False
        else:
            raise RuntimeError("Unable to process image of shape %s" % (tuple(self.shape,)))
        self.profile = bool(profile)
        self.scales = [] #in XY order
        self.procsize = []
        self.wgsize = []
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
        self._calc_workgroups
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
        """
        Nota scales are in XY order
        """
        self.scales = [tuple(numpy.int32(i) for i in self.shape[-1::-1])]
        shape = self.shape
        min_size = 2 * par.BorderDist + 2
        while min(shape) > min_size:
            shape = tuple(numpy.int32(i // 2) for i in shape)
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

        ########################################################################
        # TODO: Calculate space for gaussian kernels
        ########################################################################

    def _allocate_buffers(self):
        if self.dtype != numpy.float32:
            self.buffers["raw"] = pyopencl.array.empty(self.queue, self.shape, dtype=self.dtype, order="C")
        self.buffers["input"] = pyopencl.array.empty(self.queue, self.shape, dtype=numpy.float32, order="C")
        for octave, scale in enumerate(self.scales):
            self.buffers[(octave, "tmp") ] = pyopencl.array.empty(self.queue, scale, dtype=numpy.float32, order="C")
            for i in range(par.Scales + 3):
                self.buffers[(octave, i, "G") ] = pyopencl.array.empty(self.queue, scale, dtype=numpy.float32, order="C")
            for i in range(par.Scales + 2):
                self.buffers[(octave, i, "DoG") ] = pyopencl.array.empty(self.queue, scale, dtype=numpy.float32, order="C")
        ########################################################################
        # TODO: Allocate space for gaussian kernels
        ########################################################################

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

    def _calc_workgroups(self):
        """
        First try to guess the best workgroup size, then calculate all global worksize
        
        Nota: 
        The workgroup size is limited by the device  
        The workgroup size is limited to the 2**n below then image size (hence changes with octaves)
        The second dimension of the wg size should be large, the first small: i.e. (1,64)
        The processing size should be a multiple of  workgroup size.
        """
        device = self.ctx.devices[0]
        max_work_group_size = device.max_work_group_size
        max_work_item_sizes = device.max_work_item_sizes
        #we recalculate the shapes ...
        shape = self.shape
        min_size = 2 * par.BorderDist + 2
        while min(shape) > min_size:
            wg = (1, min(2 ** int(math.log(shape[1]) / math.log(2)), max_work_group_size[1]))
            self.wgsize.append(wg)
            self.procsize.append(calc_size(shape, wg))
            shape = tuple(i // 2 for i in shape)




    def keypoints(self, image):
        """
        Calculates the keypoints of the image
        @param image: ndimage of 2D (or 3D if RGB)  
        """
        assert image.shape == self.shape
        assert image.dtype == self.dtype
#        if image.ndim == 3 and image.shape[-1] == 3 and image.dtype==numpy.uint8: #RGB image
#            self.buffers
        t0 = time.time()
        if self.dtype == numpy.float32:
            self.buffers["input"].set(image)
        elif self.dtype == numpy.uint8:
            self.buffers["raw"].set(image)
            if self.RGB:
                self.programs["preprocess"].rgb_to_float(self.queue, self.procsize[0], self.ws[0],
                                                         self.buffers["raw"].data, self.buffers["input"].data, *self.scales[0])
            else:
                self.programs["preprocess"].u8_to_float(self.queue, self.procsize[0], self.ws[0],
                                                         self.buffers["raw"].data, self.buffers["input"].data, *self.scales[0])
        elif self.dtype == numpy.uint16:
            self.buffers["raw"].set(image)
            self.programs["preprocess"].u16_to_float(self.queue, self.procsize[0], self.ws[0],
                                                         self.buffers["raw"].data, self.buffers["input"].data, *self.scales[0])
        elif self.dtype == numpy.int32:
            self.buffers["raw"].set(image)
            self.programs["preprocess"].s32_to_float(self.queue, self.procsize[0], self.ws[0],
                                                         self.buffers["raw"].data, self.buffers["input"].data, *self.scales[0])
        elif self.dtype == numpy.int64:
            self.buffers["raw"].set(image)
            self.programs["preprocess"].s64_to_float(self.queue, self.procsize[0], self.ws[0],
                                                         self.buffers["raw"].data, self.buffers["input"].data, *self.scales[0])
        min_data = pyopencl.array.min(self.gpudata, queue).get()
        max_data = pyopencl.array.max(self.gpudata, queue).get()
        self.programs["preprocess"].normalizes(self.queue, self.procsize[0], self.ws[0],
                                               self.buffers["input"].data,
                                               numpy.float32(min_data), numpy.float32(max_data), numpy.float32(255.), *self.scales[0])

        octSize = 1.0
        curSigma = 1.0 if par.DoubleImSize else 0.5
        if par.InitSigma > curSigma:
            self.debug("Bluring image to achieve std: %f", par.InitSigma)
            sigma = math.sqrt(par.InitSigma ** 2 - curSigma ** 2)

        gaussian_convolution(image.getPlane(), image.getPlane(), image.nwidth(), image.nheight(), sigma);

        ########################################################################
        # Calculate gaussian blur and DoG for every octave
        ########################################################################
        octave = -1
        for scale, proc, wg in zip(self.scales, self.procsize, self.wgsize):
            octave += 1

