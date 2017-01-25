from setuptools import setup

setup(name='RadFil',
      version='0.1',
      description='Build Radial Density Profiles for Interstellar Filaments',
      url='http://github.com/catherinezucker/RadFil',
      author='Catherine Zucker',
      author_email='catherine.zucker@cfa.harvard.edu',
      packages=['RadFil'],
      install_requires=[
          'matplotlib',
          'networkx',
          'numpy',
          'scipy',
          'astropy',
          'lmfit'           
      ],
      test_suite='musca.collector',
      tests_require=['musca'],
      zip_safe=False)
