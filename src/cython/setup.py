# https://cython.readthedocs.io/en/latest/src/quickstart/cythonize.html
#
#  build:
#     python setup.py build_ext --inplace

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy


extensions = [
    Extension(name="polylineminer",
              sources=["polylineminer.pyx"],
              language="c++",  # extra_compile_args=["-std=c++11"],  # not necessary for windows
              include_dirs=[numpy.get_include()],
              extra_compile_args=["-std=c++11"],
              extra_link_args=[]),
]


setup(
    ext_modules=cythonize(extensions,
                          compiler_directives={"language_level": "3"}),
    include_dirs=["."],
)