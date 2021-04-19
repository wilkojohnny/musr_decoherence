from setuptools import setup
from Cython.Build import cythonize
import os

# make the c compiler clang in llvm (use this, because it allows for openmp)
# the below one is if you're on macos and are using Johnny's laptop!
# os.environ["CC"] = "/usr/local/opt/llvm/bin/clang"

setup(
    name='musr_decoherence',
    version='0.1',
    packages=['musr_decoherence'],
    install_requires=['numpy', 'scipy', 'matplotlib', 'ase', 'lmfit'],
    extras_require = {
        'gpu' : ['cupy']
    },
    ext_modules = cythonize("musr_decoherence/cython_polarisation.pyx"),
    url='',
    license='',
    author='John Wilkinson',
    author_email='john.wilkinson@physics.ox.ac.uk',
    description='musr_decoherence -- calculate and fit F--mu--F states by taking into account further nearest-neighbours.'
)
