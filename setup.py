from setuptools import setup

setup(
    name='musr_decoherence',
    version='0.1',
    packages=['musr_decoherence'],
    install_requires=['numpy', 'scipy', 'matplotlib', 'ase', 'lmfit'],
    extras_require = {
        'gpu' : ['cupy']
    },
    url='',
    license='',
    author='John Wilkinson',
    author_email='john.wilkinson@physics.ox.ac.uk',
    description='musr_decoherence -- calculate and fit F--mu--F states by taking into account further nearest-neighbours.'
)
