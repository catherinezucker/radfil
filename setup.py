from setuptools import setup

setup(
  name = 'radfil',
  packages = ['radfil'],
  version = '1.1.5',
  description = 'Build radial density profiles for interstellar filaments',
  author = ['Catherine Zucker', 'Hope Chen'],
  author_email = 'catherine.zucker@cfa.harvard.edu',
  url = 'http://github.com/catherinezucker/radfil', # use the URL to the github repo
  download_url = 'https://api.github.com/repos/catherinezucker/radfil/zipball/master',
  keywords = ['astrophysics', 'radfil', 'filaments'], # arbitrary keywords
  classifiers = [],
  install_requires=[
      'numpy',
      'scipy',
      'matplotlib',
      'astropy',
      'shapely',
      'scikit-learn',
      'scikit-image',
      'networkx',
      'pandas'
  ]
)
