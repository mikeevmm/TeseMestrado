from distutils.core import setup, Extension
from glob import glob
from os.path import dirname, realpath
import numpy as np

sources = [x for x in glob(dirname(realpath(__file__)) + '/src/*.c', recursive=True)
           if not x.endswith('main.c')] + \
          glob(dirname(realpath(__file__)) + '/qop/src/*.c', recursive=True)
py_sources = glob(dirname(realpath(__file__)) + '/python/*.c')
extras = ['-Wall', '-Wextra', '-Wconversion', '-Wno-unused-variable',
          '-Wno-unused', '-Wmissing-prototypes', '-Wstrict-prototypes', '-O3']

module = Extension('cextension', sources=sources + py_sources,
                   include_dirs=[
                       np.get_include(),
                       dirname(realpath(__file__)),
                       f"{dirname(realpath(__file__))}/qop"
                   ],
                   extra_compile_args=extras)

setup(name='cextension',
      version='1.0',
      description='Template for python C extension module.',
      ext_modules=[module])
