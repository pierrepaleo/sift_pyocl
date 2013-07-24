Usage: demo.py [options] imagefiles*

Demonstration of sift using OpenCL version C++ implementation

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit
  -d DEVICE, --device=DEVICE
                        device on which to run: coma separated like
                        --device=0,1
  -t TYPE, --type=TYPE  device type  on which to run like CPU (default) or GPU
  -p, --profile         Print profiling information of OpenExecution
  -f, --force           rebuild the package
  -r, --remove          remove build and rebuild the package
  