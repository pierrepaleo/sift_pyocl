sift_pyocl
==========

An implementation of SIFT on GPU with OpenCL


Features :

  * Can run on GPU and CPU
  * SIFT descriptors are accessible as `numpy array`
  * Helper for aligning features with affine transformation (bilinear + offset)
  * Compatible with the [ipol](http://www.ipol.im/pub/art/2015/69) implementation


## Getting started

``sift_pyocl`` provides various *plans* for the image alignment process : descriptors computation, descriptors matching and image alignment.

### Computing SIFT descriptors of an image


```python
import sift_pyocl as sift
siftPlan = sift.SiftPlan(img.shape,img.dtype,devicetype="GPU")
kp = siftp.keypoints(img)
```


### Matching the descriptors of two images

```
matchPlan = sift.MatchPlan(devicetype="GPU")
kp = siftp.match(kp1,kp2)
```

### Aligning two images

```
alignPlan = sift.LinearAlign(i1, devicetype="GPU")
i2_transformed = alignPlan.align(i2)
```