from setuptools import setup
setup(
    name='rodeo',
    version='0.4',
    author='Anonymous, Anonymous',
    author_email='anonymous@anonymous.ca',
    #packages=find_packages(exclude=["tests*", "examples"]),
    packages=["rodeo"],

    install_requires=[
        'jaxlib==0.3.2', 'jax==0.3.4', 'numpy>=1.22', 'scipy>=1.2.1', 'numba>=0.51.2', 'jupyter','matplotlib', 'seaborn',
        'diffrax'
        ],
    setup_requires=['setuptools>=38']
)
