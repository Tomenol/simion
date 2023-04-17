import setuptools

__version__ = '0.0.1'

setuptools.setup(
    name='simion',
    version=__version__,
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT',
        'Operating System :: Unix',
        'Intended Audience :: Science/Research',
    ],
    packages=setuptools.find_packages(),
    package_data={
        'simion': ['data/*'],
    },
    # metadata to display on PyPI
    author='Thomas Maynadié',
    author_email='maynadie.t@gmail.com',
    description='SIMION Helper python package',
    license='MIT',
)