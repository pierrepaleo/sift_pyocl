
"""
Contains a class for creating a plan, allocating arrays, compiling kernels and other things like that
"""
import time, math, os, logging
import gc
import numpy
import pyopencl, pyopencl.array
from .param import par
from .opencl import ocl
from .utils import calc_size, kernel_size
logger = logging.getLogger("sift.plan")


class SiftPlan(object):
    """
    How to calculate a set of SIFT keypoint on an image:

    siftp = sift.SiftPlan(img.shape,img.dtype,devicetype="GPU")
    kp = siftp.keypoints(img)

    kp is a nx132 array. the second dimension is composed of x,y, scale and angle as well as 128 floats describing the keypoint

    """
    kernels = ["convolution", "preprocess", "algebra", "image"]
    converter = {numpy.dtype(numpy.uint8):"u8_to_float",
                 numpy.dtype(numpy.uint16):"u16_to_float",
                 numpy.dtype(numpy.int32):"s32_to_float",
                 numpy.dtype(numpy.int64):"s64_to_float",
#                    numpy.float64:"double_to_float",
                      }
    sigmaRatio = 2.0 ** (1.0 / par.Scales)
    PIX_PER_KP = 10 # pre_allocate buffers for keypoints

    def __init__(self, shape=None, dtype=None, devicetype="GPU", template=None, profile=False, device=None, PIX_PER_KP=None):
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
        if PIX_PER_KP :
            self.PIX_PER_KP = int(PIX_PER_KP)
        self.profile = bool(profile)
        self.scales = [] #in XY order
        self.procsize = []
        self.wgsize = []
        self.kpsize = None
        self.buffers = {}
        self.programs = {}
        self.memory = None
        self.octave_max = None
        self._calc_scales()
        self._calc_memory()
        if device is None:
            self.device = ocl.select_device(type=devicetype, memory=self.memory, best=True)
        else:
            self.device = device
        self.ctx = ctx = pyopencl.Context(devices=[pyopencl.get_platforms()[self.device[0]].get_devices()[self.device[1]]])
        print self.ctx.devices[0]
        if profile:
            self.queue = pyopencl.CommandQueue(self.ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
        else:
            self.queue = pyopencl.CommandQueue(self.ctx)
        self._calc_workgroups()
        self._compile_kernels()
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
        """
        Nota scales are in XY order
        """
        self.scales = [tuple(numpy.int32(i) for i in self.shape[-1::-1])]
        shape = self.shape
        min_size = 2 * par.BorderDist + 2
        while min(shape) > min_size * 2:
            shape = tuple(numpy.int32(i // 2) for i in shape)
            self.scales.append(shape)
#        self.scales.pop()
        self.octave_max = len(self.scales)

    def _calc_memory(self):
        # Just the context + kernel takes about 75MB on the GPU
        self.memory = 75 * 2 ** 20
        size_of_float = numpy.dtype(numpy.float32).itemsize
        size_of_input = numpy.dtype(self.dtype).itemsize
        #raw images:
        size = self.shape[0] * self.shape[1]
        self.memory += size * (size_of_float + size_of_input) #raw_float + initial_image
        if self.RGB:
            self.memory += 2 * size * (size_of_input) # one of three was already counted
        for scale in self.scales:
            nr_blur = 3 #one input, one output and one tmp
            nr_dogs = par.Scales + 2
            size = scale[0] * scale[1]
            self.memory += size * (nr_blur + nr_dogs) * size_of_float
        self.kpsize = int(self.shape[0] * self.shape[1] // self.PIX_PER_KP)   # Is the number of kp independant of the octave ? int64 causes problems with pyopencl
        self.memory += self.kpsize * size_of_float * 4 * 2 # those are array of float4 to register keypoints, we need two of them
        self.memory += 4 #keypoint index Counter


        ########################################################################
        # Calculate space for gaussian kernels
        ########################################################################
        curSigma = 1.0 if par.DoubleImSize else 0.5
        if par.InitSigma > curSigma:
            sigma = math.sqrt(par.InitSigma ** 2 - curSigma ** 2)
            size = kernel_size(sigma, True)
            # TODO: possible enhancement, if size is even make it odd
            logger.debug("pre-Allocating %s float for init blur" % size)
            self.memory += size * size_of_float
        prevSigma = par.InitSigma
        for i in range(par.Scales + 2):
            increase = prevSigma * math.sqrt(self.sigmaRatio ** 2 - 1.0)
            size = kernel_size(increase, True)
            logger.debug("pre-Allocating %s float for blur sigma: %s" % (size, increase))
            self.memory += size * size_of_float
            prevSigma *= self.sigmaRatio;

    def _allocate_buffers(self):
        shape = self.shape
        if self.dtype != numpy.float32:
            if self.RGB:
                rgbshape = self.shape[0], self.shape[1], 3
                self.buffers["raw"] = pyopencl.array.empty(self.queue, rgbshape, dtype=self.dtype, order="C")
            else:
                self.buffers["raw"] = pyopencl.array.empty(self.queue, shape, dtype=self.dtype, order="C")
        self.buffers["input"] = pyopencl.array.empty(self.queue, shape, dtype=numpy.float32, order="C")
        self.buffers[ "Kp_1" ] = pyopencl.array.empty(self.queue, (self.kpsize, 4), dtype=numpy.float32, order="C")
        self.buffers[ "Kp_2" ] = pyopencl.array.empty(self.queue, (self.kpsize, 4), dtype=numpy.float32, order="C")
        self.buffers["cnt" ] = pyopencl.array.empty(self.queue, (1,), dtype=numpy.int32, order="C")

        for octave in range(self.octave_max):
            self.buffers[(octave, "tmp") ] = pyopencl.array.empty(self.queue, shape, dtype=numpy.float32, order="C")
            self.buffers[(octave, "G_1") ] = pyopencl.array.empty(self.queue, shape, dtype=numpy.float32, order="C")
            self.buffers[(octave, "G_2") ] = pyopencl.array.empty(self.queue, shape, dtype=numpy.float32, order="C")
            self.buffers[(octave, "DoGs") ] = pyopencl.array.empty(self.queue, (par.Scales + 2, shape[0], shape[1]),
                                                                   dtype=numpy.float32, order="C")
            shape = (shape[0] // 2, shape[1] // 2)
        for buffer in self.buffers.values():
            buffer.fill(0)
        ########################################################################
        # Allocate space for gaussian kernels
        ########################################################################
        curSigma = 1.0 if par.DoubleImSize else 0.5
        if par.InitSigma > curSigma:
            sigma = math.sqrt(par.InitSigma ** 2 - curSigma ** 2)
            self._init_gaussian(sigma)
        prevSigma = par.InitSigma

        for i in range(par.Scales + 2):
            increase = prevSigma * math.sqrt(self.sigmaRatio ** 2 - 1.0)
            self._init_gaussian(increase)
            prevSigma *= self.sigmaRatio


    def _init_gaussian(self, sigma):
        """
        Create a buffer of the right size according to the width of the gaussian ...


        @param  sigma: width of the gaussian, the length of the function will be 8*sigma + 1

        Same calculation done on CPU
        x = numpy.arange(size) - (size - 1.0) / 2.0
        gaussian = numpy.exp(-(x / sigma) ** 2 / 2.0).astype(numpy.float32)
        gaussian /= gaussian.sum(dtype=numpy.float32)
        """
        name = "gaussian_%s" % sigma
        size = kernel_size(sigma, True)
        logger.debug("Allocating %s float for blur sigma: %s" % (size, sigma))
        gaussian_gpu = pyopencl.array.empty(self.queue, size, dtype=numpy.float32, order="C")
#       Norming the gaussian takes three OCL kernel launch (gaussian, calc_sum and norm) -
        self.programs["preprocess"].gaussian(self.queue, (size,), (1,),
                                            gaussian_gpu.data,         #__global     float     *data,
                                            numpy.float32(sigma),      #const        float     sigma,
                                            numpy.int32(size))         #const        int     SIZE
        sum_data = pyopencl.array.sum(gaussian_gpu, dtype=numpy.float32, queue=self.queue).get()
        self.programs["preprocess"].divide_cst(self.queue, (size,), (1,),
                                              gaussian_gpu.data,         #__global     float     *data,
                                              sum_data,                  #const        float     sigma,
                                              numpy.int32(size))         #const        int     SIZE

        self.buffers[name] = gaussian_gpu


    def _free_buffers(self):
        """
        free all memory allocated on the device
        """
        for buffer_name in self.buffers:
            if self.buffers[buffer_name] is not None:
                try:
                    del self.buffers[buffer_name]
                    self.buffers[buffer_name] = None
                except pyopencl.LogicError:
                    logger.error("Error while freeing buffer %s" % buffer_name)

    def _compile_kernels(self):
        """
        Call the OpenCL compiler
        """
        for kernel in self.kernels:
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
            wg = (1, min(2 ** int(math.log(shape[1]) / math.log(2)), max_work_item_sizes[1]))
            self.wgsize.append(wg)
            self.procsize.append(calc_size(shape, wg))
            shape = tuple(i // 2 for i in shape)




    def keypoints(self, image):
        """
        Calculates the keypoints of the image
        @param image: ndimage of 2D (or 3D if RGB)
        """
        total_size = 0
        keypoints = []
        descriptors = []
        assert image.shape[:2] == self.shape
        assert image.dtype == self.dtype
        t0 = time.time()

        if self.dtype == numpy.float32:
            pyopencl.enqueue_copy(self.queue, self.buffers["input"].data, image)
        elif (image.ndim == 3) and (self.dtype == numpy.uint8) and (self.RGB):
            pyopencl.enqueue_copy(self.queue, self.buffers["raw"].data, image)
            self.programs["preprocess"].rgb_to_float(self.queue, self.procsize[0], self.wgsize[0],
                                                         self.buffers["raw"].data, self.buffers["input"].data, *self.scales[0])

        elif self.dtype in self.converter:
            program = self.programs["preprocess"].__getattr__(self.converter[self.dtype])
            pyopencl.enqueue_copy(self.queue, self.buffers["raw"].data, image)
            program(self.queue, self.procsize[0], self.wgsize[0],
                    self.buffers["raw"].data, self.buffers["input"].data, *self.scales[0])
        else:
            raise RuntimeError("invalid input format error")
        min_data = pyopencl.array.min(self.buffers["input"], self.queue).get()
        max_data = pyopencl.array.max(self.buffers["input"], self.queue).get()
        self.programs["preprocess"].normalizes(self.queue, self.procsize[0], self.wgsize[0],
                                               self.buffers["input"].data,
                                               numpy.float32(min_data), numpy.float32(max_data), numpy.float32(255.), *self.scales[0])


        octSize = 1.0
        curSigma = 1.0 if par.DoubleImSize else 0.5
        octave = 0
        if par.InitSigma > curSigma:
            logger.debug("Bluring image to achieve std: %f", par.InitSigma)
            sigma = math.sqrt(par.InitSigma ** 2 - curSigma ** 2)
            self._gaussian_convolution(self.buffers["input"], self.buffers[(0, "G_1")], sigma, 0)
        else:
            pyopencl.enqueue_copy(self.queue, dest=self.buffers[(0, "G_1")].data, src=self.buffers["input"].data)

        ########################################################################
        # Rescale all images to populate all octaves
        ########################################################################
        for octave in range(self.octave_max - 1):
             self.programs["preprocess"].shrink(self.queue, self.procsize[octave + 1], self.wgsize[octave + 1],
                                                self.buffers[(octave, "G_1")].data, self.buffers[(octave + 1, "G_1")].data,
                                                numpy.int32(2), numpy.int32(2), *self.scales[octave + 1])


        for octave in range(self.octave_max):
            prevSigma = par.InitSigma
            logger.debug("Calculating DoGs on octave %i" % octave)
            for i in range(par.Scales + 2):
                sigma = prevSigma * math.sqrt(self.sigmaRatio ** 2 - 1.0)
                logger.debug("blur with sigma %s" % sigma)
                ########################################################################
                # Calculate gaussian blur and DoG for every octave
                ########################################################################

                self._gaussian_convolution(self.buffers[(octave, "G_1")], self.buffers[(octave, "G_2")], sigma, octave)
                prevSigma *= self.sigmaRatio
                self.programs["algebra"].combine(self.queue, self.procsize[octave], self.wgsize[octave],
                                                 self.buffers[(octave, "G_2")].data, numpy.float32(1.0),
                                                 self.buffers[(octave, "G_1")].data, numpy.float32(-1.0),
                                                 self.buffers[(octave, "DoGs")].data, numpy.int32(i),
                                                 *self.scales[octave])
                #swap buffers:
                self.buffers[(octave, "G_1")], self.buffers[(octave, "G_2")] = self.buffers[(octave, "G_2")], self.buffers[(octave, "G_1")]

                #recycle buffer G_2 and tmp to store ori and grad
                ori = self.buffers[(octave, "G_2")]
                grad = self.buffers[(octave, "tmp")]
                self.programs["image"].compute_gradient_orientation(self.queue, self.procsize[octave], self.wgsize[octave],
                           self.buffers[(octave, "G_1")].data, #__global float* igray,
                           grad.data,                          #__global float *grad,
                           ori.data,                           #__global float *ori,
                           *self.scales[octave])               #int width,int height

#THIS WHOOL BLOCK needs to be indented ! TODO

        ########################################################################
        # Calculate Keypoints per octave
        ########################################################################
        wgsize = (8,)#(max(self.wgsize[octave]),) #TODO: optimize
        kpsize32 = numpy.int32(self.kpsize)
        for octave in range(self.octave_max):
            logger.debug("Calculating keypoints on octave %i" % octave)
            self._reset_keypoints()
            octsize = numpy.int32(2 ** octave)
            for scale in range(1, par.Scales + 1):
                print (octave, scale)
                self.programs["image"].local_maxmin(self.queue, self.procsize[octave], self.wgsize[octave],
                                                    self.buffers[(octave, "DoGs")].data,            #__global float* DOGS,
                                                    self.buffers["Kp_1"].data,                      #__global keypoint* output,
                                                    numpy.int32(par.BorderDist),                    #int border_dist,
                                                    numpy.float32(par.PeakThresh),                  #float peak_thresh,
                                                    octsize,  # int octsize,
                                                    numpy.float32(par.EdgeThresh1),                 #float EdgeThresh0,
                                                    numpy.float32(par.EdgeThresh),                  #float EdgeThresh,
                                                    self.buffers["cnt"].data,                       #__global int* counter,
                                                    kpsize32,                                       #int nb_keypoints,
                                                    numpy.int32(scale),                             #int scale,
                                                    *self.scales[octave])                           #int width, int height)


                procsize = calc_size((self.kpsize,), wgsize)
    #           Refine keypoints
                kp_counter = self.buffers["cnt"].get()[0]
                if kp_counter > 0.9 * self.kpsize:
                    logger.warning("Keypoint counter overflow risk: counted %s / %s" % (kp_counter, self.kpsize))
                print("Keypoint counted %s / %s" % (kp_counter, self.kpsize))
                self.programs["image"].interp_keypoint(self.queue, procsize, wgsize,
                                              self.buffers[(octave, "DoGs")].data,  #__global float* DOGS,
                                              self.buffers["Kp_1"].data,            # __global keypoint* keypoints,
                                              kp_counter,                           # int actual_nb_keypoints,
                                              numpy.float32(par.PeakThresh),        # float peak_thresh,
                                              numpy.float32(par.InitSigma),         # float InitSigma,
                                              *self.scales[octave])                 # int width, int height)
                self.buffers["cnt"].set(numpy.array([0], dtype=numpy.int32))
#           Orientation assignement: 1D kernel, rather heavy kernel
                procsize = calc_size((newcnt,), wgsize)
                self.programs["image"].orientation_assignment(procsize, wgsize,
                                      self.buffers["Kp_1"].data,  # __global keypoint* keypoints,
                                                                #__global float* grad,
                                                                #__global float* ori,
                                         self.buffers["cnt"].data,  # __global int* counter,
                                         octsize,               #int octsize,
                                         numpy.float32(par.OriSigma),  # float OriSigma, //WARNING: (1.5), it is not "InitSigma (=1.6)"
                                         kpsize32,  # int max of nb_keypoints,
                                         newcnt,                #int actual_nb_keypoints,
                                         self.scales[octave])   #int grad_width, int grad_height)

                self.programs["algebra"].compact(self.queue, procsize, wgsize,
                                self.buffers["Kp_1"].data, # __global keypoint* keypoints,
                                self.buffers["Kp_2"].data, #__global keypoint* output,
                                self.buffers["cnt"].data,  #__global int* counter,
                                kpsize32                   #int nbkeypoints
                                                 )
                newcnt = self.buffers["cnt"].get()[0]
                print("After compaction, %i (-%i)" % (newcnt, kp_counter - newcnt))

                #swap keypoints:
                self.buffers["Kp_1"], self.buffers["Kp_2"] = self.buffers["Kp_2"], self.buffers["Kp_1"]

#                compact again ?
#                generate keypoints ?
#           transfer to CPU
            kp_counter = self.buffers["cnt"].get()[0]
            keypoints.append(self.buffers["Kp_1"].get()[:kp_counter])
            total_size += kp_counter

        ########################################################################
        # Merge keypoints in central memory
        ########################################################################
        output = numpy.zeros((total_size, 4), dtype=numpy.float32)
        last = 0
        for ds in keypoints:
            l = ds.shape[0]
            ds[last:last + l] = ds
            last += l
        print("Execution time: %.3fs" % (time.time() - t0))
#        self.count_kp(output)
        return output

    def _gaussian_convolution(self, input_data, output_data, sigma, octave=0):
        """
        Calculate the gaussian convolution with precalculated kernels.

        Uses a temporary buffer
        """
        temp_data = self.buffers[(octave, "tmp") ]
        gaussian = self.buffers["gaussian_%s" % sigma]
        k1 = self.programs["convolution"].horizontal_convolution(self.queue, self.procsize[octave], self.wgsize[octave],
                                input_data.data, temp_data.data, gaussian.data, numpy.int32(gaussian.size), *self.scales[octave])
        k2 = self.programs["convolution"].vertical_convolution(self.queue, self.procsize[octave], self.wgsize[octave],
                                temp_data.data, output_data.data, gaussian.data, numpy.int32(gaussian.size), *self.scales[octave])
        if self.profile:
            k2.wait()
            logger.info("Blur sigma %s octave %s took %.3fms + %.3fms" % (sigma, octave, 1e-6 * (k1.profile.end - k1.profile.start),
                                                                                          1e-6 * (k2.profile.end - k2.profile.start)))

    def _reset_keypoints(self):
        self.buffers["Kp_1"].fill(-1, self.queue)
        self.buffers["Kp_2"].fill(-1, self.queue)
        self.buffers["cnt"].fill(0, self.queue)

    def count_kp(self,output):
        kpt = 0
        for octave, data in enumerate(output):
#            if octave % 2 == 0:
#                continue
#            octave /= 2
            sum = (data[:, 1] != -1.0).sum()
            kpt += sum
            print("octave %i kp count %i/%i size %s ratio:%s" % (octave, sum, self.kpsize, self.scales[octave], 1000.0 * sum / self.scales[octave][1] / self.scales[octave][0]))
        print("Found total %i guess %s pixels per keypoint" % (kpt, self.shape[0] * self.shape[1] / kpt))
if __name__ == "__main__":
    #Prepare debugging
    import scipy.misc
    lena = scipy.lena()
    s = SiftPlan(template=lena)
    s.keypoints(lena)

