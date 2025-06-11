from setuptools import setup, find_packages

setup(
    name='ADCPySpec',
    version='0.1.0',
    description='A Python package for spectral analysis of ADCP and velocity field data.',
    author='Cristina Martí',
    author_email='cmarti@imedea.uib-csic.es',
    url='https://github.com/cmartisolana/ADCPySpec',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pytest'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    python_requires='>=3.6',
    include_package_data=True,
)