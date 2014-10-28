#!/usr/bin/env python
#-*- coding: utf-8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/kif/sift_pyocl
#

"""
Installer script for SIFT algorithm in PyOpenCL 
"""

from __future__ import division, with_statement, print_function


__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "2014-10-28"
__status__ = "stable"
__license__ = """
Permission is hereby granted, free of charge, to any person
obtaining a copy of ethis software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

"""



import os, sys, glob, shutil, ConfigParser, platform
from distutils.core import setup, Extension, Command
from numpy.distutils.misc_util import get_numpy_include_dirs
from distutils.command.install_data import install_data
from distutils.sysconfig import get_python_lib

# ###############################################################################
# Check for Cython
# ###############################################################################
try:
    from Cython.Distutils import build_ext
    CYTHON = True
except ImportError:
    CYTHON = False
if CYTHON:
    try:
        import Cython.Compiler.Version
    except ImportError:
        CYTHON = False
    else:
        if Cython.Compiler.Version.version < "0.17":
            CYTHON = False

if CYTHON:
    cython_c_ext = ".pyx"
else:
    cython_c_ext = ".c"
    from distutils.command.build_ext import build_ext
cmdclass = {}


def rewriteManifest(with_testimages=False):
    """
    Rewrite the "Manifest" file ... if needed

    @param with_testimages: include
    """
    base = os.path.dirname(os.path.abspath(__file__))
    manifest_file = os.path.join(base, "MANIFEST.in")
    if not os.path.isfile(manifest_file):
        print("MANIFEST file is missing !!!")
        return
    manifest = [i.strip() for i in open(manifest_file)]
    changed = False

    if with_testimages:
        testimages = ["test/testimages/" + i for i in os.listdir(os.path.join(base, "test", "testimages"))]
        for image in testimages:
            if image not in manifest:
                manifest.append("include " + image)
                changed = True
    else:
        for line in manifest[:]:
            if line.startswith("include test/testimages"):
                changed = True
                manifest.remove(line)
    if changed:
        with open(manifest_file, "w") as f:
            f.write(os.linesep.join(manifest))
        # remove MANIFEST: will be re generated !
        os.unlink(manifest_file[:-3])

if ("sdist" in sys.argv):
    if ("--with-testimages" in sys.argv):
        sys.argv.remove("--with-testimages")
        rewriteManifest(with_testimages=True)
    else:
        rewriteManifest(with_testimages=False)


installDir = "sift" #relative to site-packages ...
data_files = [(installDir, glob.glob("openCL/*.cl"))]


class smart_install_data(install_data):
    def run(self):
        install_cmd = self.get_finalized_command('install')
        self.install_dir = getattr(install_cmd, 'install_lib')
        print("DATA to be installed in %s" % self.install_dir)
        return install_data.run(self)
cmdclass['install_data'] = smart_install_data


if sys.platform == "win32":
    # This is for mingw32/gomp?
#    data_files[0][1].append(os.path.join("dll", "pthreadGC2.dll"))
    root = os.path.dirname(os.path.abspath(__file__))
    tocopy_files = []
    script_files = []
    for i in os.listdir(os.path.join(root, "scripts")):
        if os.path.isfile(os.path.join(root, "scripts", i)):
            if i.endswith(".py"):
                script_files.append(os.path.join("scripts", i))
            else:
                tocopy_files.append(os.path.join("scripts", i))
    for i in tocopy_files:
        filein = os.path.join(root, i)
        if (filein + ".py") not in script_files:
            shutil.copyfile(filein, filein + ".py")
            script_files.append(filein + ".py")

else:
    script_files = glob.glob("scripts/*")

version = [eval(l.split("=")[1]) for l in open(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "sift-src", "__init__.py"))
    if l.strip().startswith("version")][0]


# We subclass the build_ext class in order to handle compiler flags
# for openmp and opencl etc in a cross platform way
translator = {
        # Compiler
            # name, compileflag, linkflag
        'msvc' : {
            'openmp' : ('/openmp', ' '),
            'debug'  : ('/Zi', ' '),
            'OpenCL' : 'OpenCL',
            },
        'mingw32':{
            'openmp' : ('-fopenmp', '-fopenmp'),
            'debug'  : ('-g', '-g'),
            'stdc++' : 'stdc++',
            'OpenCL' : 'OpenCL'
            },
        'default':{
            'openmp' : ('-fopenmp', '-fopenmp'),
            'debug'  : ('-g', '-g'),
            'stdc++' : 'stdc++',
            'OpenCL' : 'OpenCL'
            }
        }


class build_ext_sift(build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type in translator:
            trans = translator[self.compiler.compiler_type]
        else:
            trans = translator['default']

        for e in self.extensions:
            e.extra_compile_args = [ trans[a][0] if a in trans else a
                                    for a in e.extra_compile_args]
            e.extra_link_args = [ trans[a][1] if a in trans else a
                                 for a in e.extra_link_args]
            e.libraries = filter(None, [ trans[a] if a in trans else None
                                        for a in e.libraries])

            # If you are confused look here:
            # print e, e.libraries
            # print e.extra_compile_args
            # print e.extra_link_args
        build_ext.build_extensions(self)
cmdclass['build_ext'] = build_ext_sift

class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import sys, subprocess
        os.chdir("test")
        errno = subprocess.call([sys.executable, 'test_all.py'])
        if errno != 0:
            print("Tests did not pass !!!")
            #raise SystemExit(errno)
        else:
            print("All Tests passed")
        os.chdir("..")
cmdclass['test'] = PyTest

#######################
# build_doc commandes #
#######################

try:
    import sphinx
    import sphinx.util.console
    sphinx.util.console.color_terminal = lambda: False
    from sphinx.setup_command import BuildDoc
except ImportError:
    sphinx = None

if sphinx:
    class build_doc(BuildDoc):

        def run(self):

            # make sure the python path is pointing to the newly built
            # code so that the documentation is built on this and not a
            # previously installed version

            build = self.get_finalized_command('build')
            sys.path.insert(0, os.path.abspath(build.build_lib))

            # Build the Users Guide in HTML and TeX format
            for builder in ('html', 'latex'):
                self.builder = builder
                self.builder_target_dir = os.path.join(self.build_dir, builder)
                self.mkpath(self.builder_target_dir)
                builder_index = 'index_{0}.txt'.format(builder)
                BuildDoc.run(self)
            sys.path.pop(0)
    cmdclass['build_doc'] = build_doc

classifiers = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Programming Language :: Python
Topic :: Software Development :: Libraries :: Python Modules
Operating System :: Microsoft :: Windows
Operating System :: Unix
Operating System :: MacOS :: MacOS X
Operating System :: POSIX

"""


setup(name='sift_pyocl',
      version=version,
      author="Pierre Paleo, Jérôme Kieffer",
      author_email="jerome.kieffer@esrf.fr",
      description='Python/OpenCL implementation of Sift algorithm image alignment',
      url="https://github.com/kif/sift_pyocl",
      download_url="https://github.com/kif/sift_pyocl/archive/master.zip",
      scripts=script_files,
#      ext_package="sift",
#      ext_modules=[Extension(**dico) for dico in ext_modules],
      packages=["sift_pyocl"],
      package_dir={"sift_pyocl": "sift-src" },
      test_suite="test",
      cmdclass=cmdclass,
      data_files=data_files,
      classifiers=filter(None, classifiers.split("\n")),
      license="MIT"
      )



#print(data_files)
try:
    import pyopencl
except ImportError:
    print("""sift can use pyopencl to run on parallel accelerators like GPU; this is not optional !!!.
This python module can be found on:
http://pypi.python.org/pypi/pyopencl
""")



