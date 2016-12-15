# sift_pyocl


`sift_pyocl` is an implementation of SIFT algorithm on GPU, in OpenCL programming language.

The documentation can be found on the page of [the SILX project](http://www.silx.org/doc/silx/dev/Tutorials/Sift/sift.html).


Features :

  * Can run on GPU and CPU
  * SIFT descriptors are accessible as `numpy array`
  * Helper for images features with affine transformation
  * Compatible with the [ipol](http://www.ipol.im/pub/art/2015/69) implementation


## Important note

This module is now part of the [SILX project](https://github.com/silx-kit/silx) (`silx.image.sift`). 
All the maintenance efforts are focused in this repository.
The current repository will nevertheless be occasionally updated, in order to provide an independent SIFT module.


## Installation
Once downloaded, the module can be installed with
```bash
python setup.py
```
or locally with `python setup.py install --user`.


## Getting started

``sift_pyocl`` provides various *plans* for the image alignment process : descriptors computation, descriptors matching and image alignment. For more informations, please refer to the  [documentation](http://www.silx.org/doc/silx/dev/Tutorials/Sift/sift.html).


### Computing SIFT descriptors of an image

The `SiftPlan` object provides a plan for computing the SIFT descriptors of a given image. The plan is defined for a given image shape (and target device). Returning the SIFT descriptors from an image gives an extended flexibility for further processing (outliers removal, alignment, classification).


```python
import sift_pyocl as sift
siftPlan = sift.SiftPlan(img.shape, img.dtype, devicetype="GPU")
kp = siftp.keypoints(img)
```


### Matching the descriptors of two images

The `MatchPlan` objects provides a plan for matching the descriptors of two images.

```
matchPlan = sift.MatchPlan(devicetype="GPU")
kp = siftp.match(kp1,kp2)
```

### Aligning two images

The `LinearAlign` class provides a plan for aligning images deformed with linear transformations (translation, rotation, scaling, shear).

```
alignPlan = sift.LinearAlign(img1, devicetype="GPU")
img2_aligned = alignPlan.align(img2)
```
