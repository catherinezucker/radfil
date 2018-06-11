# RadFil
RadFil is a radial density profile building and fitting tool for interstellar filaments. All you need to build and fit a radial density profile for your own filaments is an image array and (in most cases) a boolean mask array delineating the boundary of your filament. RadFil can do the rest! Please see the tutorial (housed in RadFil_Tutorial.ipynb) for a complete working example of the code. 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1172202.svg)](https://doi.org/10.5281/zenodo.1172202)

Python2 vs. Python3
------------
*   RadFil is cross-compatible between Python 2.7, Python 3.4, Python 3.5, and Python 3.6. If you do not already have a pre-built spine and want to make one, make sure you have at least version 1.6 of fil_finder in order to ensure cross-compatibility. 

Installation
------------

RadFil can be installed via pip:

```
pip install radfil
```

To upgrade:

```
pip install --upgrade radfil
```

To install from the repository, download the zip file from github and run the following in the top level of the directory:
```
python setup.py install
```

Package Dependencies
--------------------

Requires:

 *   numpy
 *   scipy
 *   matplotlib
 *   astropy
 *   shapely
 *   scikit-learn
 *   scikit-image
 *   networkx
 *   pandas


Optional:
 *  <a href="https://github.com/e-koch/FilFinder">fil-finder</a> -- only required if you are *not* inputing a precomputed filament spine, and you want RadFil to use the fil_finder package to create one for you
 *   <a href="https://pypi.python.org/pypi/descartes">descartes</a>  -- only required if you do *not* input a filament mask, and you still want to shift the profile along each cut to the pixel with the peak column density. If you have conda, you can install descartes using the command "conda install -c conda-forge descartes"
 
 Questions? Comments?
--------------------
The RadFil package has been developed by Catherine Zucker (catherine.zucker@cfa.harvard.edu) and Hope Chen (hhchen@cfa.harvard.edu). If you find a bug, have questions, or would like to request a new feature, please feel free to send us an email or raise an issue in the github repository. We'd love to hear from you!
