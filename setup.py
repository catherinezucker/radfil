from distutils.core import setup

setup(
  name = 'RadFil',
  packages = ['RadFil'],
  version = '0.1',
  description = 'Build radial density profiles for interstellar filaments',
  author = ['Catherine Zucker', 'Hope Chen'],
  author_email = ['catherine.zucker@cfa.harvard.edu', 'hhchen@cfa.harvard.edu'],
  url = 'http://github.com/catherinezucker/RadFil', # use the URL to the github repo
  download_url = 'http://github.com/catherinezucker/RadFil/archive/0.1.tar.gz',
  keywords = ['astrophysics', 'radfil', 'filaments'], # arbitrary keywords
  classifiers = [],
  install_requires=[
      'numpy',
      'scipy',
      'matplotlib',
      'astropy',
      'shapely',
      'sklearn',
      'skimage',
      'networkx'
  ]
)
