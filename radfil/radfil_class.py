import sys
import warnings

import numpy as np
import numbers
import math
from collections import defaultdict
import types
from scipy.interpolate import splprep
from scipy.interpolate import splev
import matplotlib.pyplot as plt
import matplotlib as mpl

import astropy.units as u
import astropy.constants as c
from astropy.modeling import models, fitting, polynomial
from astropy.stats import sigma_clip
from astropy.io import fits
from fil_finder import fil_finder_2D
import shapely.geometry as geometry

from radfil import profile_tools
from plummer import Plummer1D
from matplotlib.patches import Rectangle

mpl.rcParams['grid.color'] = 'yellow'



class radfil(object):

    """
    Container object which stores the required metadata for building the radial profiles

    Parameters
    ------
    image : numpy.ndarray
        A 2D array of the data to be analyzed.

    mask: numpy.ndarray
        A 2D array defining the shape of the filament; must be of boolean
        type and the same shape as the image array
        
    beamwidth: float
        A float in units of arcseconds indicating the beamwidth of the image
        array.  
        
    header : astropy.io.fits.Header
        The header corresponding to the image array

    distance : a number-like object
        Distance to the filament; must be entered in pc

    filspine: numpy.ndarray, optional
        A 2D array defining the longest path through the filament mask; must
        be of boolean type and the same shape as the img array. Can also create
        your own with the FilFinder package using the "make_fil_spine" method.

    padsize: {sequence, array_like, int}, optional
        In cases in which the filament is too close to the border of the image,
        might need to pad in order for RadFil to successfully run.

        This is passed on to `numpy.pad`. Number of values padded to the edges
        of each axis. ((before_1, after_1), ... (before_N, after_N)) unique pad
        widths for each axis. ((before, after),) yields same before and after
        pad for each axis. (pad,) or int is a shortcut for before = after = pad
        width for all axes.

    imgscale: float, optional
        In cases where the header is not in the standrad format, imgscale is
        specified.  This is overwritten when the header and proper header keys
        exist.

    Attributes
    ----------
        imgscale : float
           The image scale in pc of each pixel
    """

    def __init__(self, image, mask, header = None, distance = None, filspine=None, padsize=None, imgscale=None):

        # Read image
        if (isinstance(image, np.ndarray)) and (image.ndim == 2):
            self.image = image
        else:
            raise TypeError("The input `image` has to be a 2D numpy array.")

        # Read mask
        if (isinstance(mask, np.ndarray)) and (mask.ndim == 2):
            self.mask = (mask & np.isfinite(self.image))
        else:
            raise TypeError("The input `mask` has to be a 2d numpy array.")

        # Read header
        if (isinstance(header, fits.header.Header)):
            self.header = header
            if ("CDELT1" in self.header.keys()) and (abs(self.header["CDELT1"]) == abs(self.header["CDELT2"])):
                self.imgscale_ang = abs(header["CDELT1"])*u.deg # degrees
            elif ("CD1_1" in self.header.keys()) and (abs(self.header["CD1_1"]) == abs(self.header["CD2_2"])):
                self.imgscale_ang = abs(header["CD1_1"])*u.deg # degrees
        else:
            self.header = None
            self.distance = None
            self.imgscale_ang = None
            warnings.warn("`header` and `distance` will not be used; all calculations in pixel units.")

        # Read distance
        ## `self.distance` is in pc.
        if isinstance(distance, numbers.Number):
            self.distance = float(distance) * u.pc
        ## if distance is wrong or None, calculate in pixel units.
        else:
            self.distance = None
            self.header = None
            warnings.warn("`header` and `distance` will not be used; all calculations in pixel units.")


        # Read filspine/define filspine
        if (isinstance(filspine, np.ndarray) and (filspine.ndim == 2)) or (filspine is None):
            self.filspine = filspine

            # calculate "imgscale" when the spine is provided. This saves users
            # the trouble to run `make_fil_spine`.
            if (isinstance(filspine, np.ndarray) and (filspine.ndim == 2)):
                # Calculate pixel scale ("imgscale"), in the unit of pc/Deal with non-standard fits header
                if (self.header is not None):
                    if ("CDELT1" in self.header.keys()) and (abs(self.header["CDELT1"]) == abs(self.header["CDELT2"])):
                        # `imgscale` in u.pc
                        ## The change to u.pc has not been cleaned up for the rest of the code, yet.
                        self.imgscale = abs(header["CDELT1"]) * (np.pi / 180.0) * self.distance
                    elif ("CD1_1" in self.header.keys()) and (abs(self.header["CD1_1"]) == abs(self.header["CD2_2"])):
                        # `imgscale` in u.pc
                        self.imgscale = abs(header["CD1_1"]) * (np.pi / 180.0) * self.distance
                    else:
                        if isinstance(imgscale, numbers.Number):
                            self.imgscale = float(imgscale) * u.pc
                            warnings.warn("The keyword `imgscale`, instead of the header, is used in calculations of physical distances.")
                        else:
                            self.imgscale = 1. * u.pix
                            warnings.warn("Calculate in pixel scales.")
                ##
                else:
                    self.imgscale = 1. * u.pix
                    warnings.warn("Calculate in pixel scales.")
            ##
            # When there is not `filspine` in the input.
            else:
                self.fispine = None
                self.imgscale = None

        else:
            self.filspine = None
            self.imgscale = None
            warnings.warn("The input `filspine` has to be a 2D numpy array. Ignore for now.")


        # Pad the edge when padsize is given.
        ## TypeError is dealt with by `numpy.pad`.
        self.padsize = padsize
        if (self.padsize is not None):
            self.image = np.pad(self.image,
                                self.padsize,
                                'constant',
                                constant_values = 0)
            self.mask = np.pad(self.mask,
                               self.padsize,
                               'constant',
                               constant_values = 0)
            if (self.filspine is not None):
                self.filspine=np.pad(self.filspine,
                                     self.padsize,
                                     'constant',
                                     constant_values = 0)


        # Return a dictionary to store the key setup Parameters
        ## (all padded)
        params = {'image': self.image,
                  'mask': self.mask,
                  'header': self.header,
                  'distance': self.distance,
                  'padsize': self.padsize,
                  'imgscale': self.imgscale}
        self._params = {'__init__': params}

        # Return a dictionary to store the results.
        self._results = {'make_fil_spine': {'filspine': self.filspine}}


    def make_fil_spine(self,beamwidth=None,verbose=False):

        """
        Create filament spine using the FilFinder package 'shortest path' option

        Parameters:
         ----------

        verbose: boolean
            A boolean indicating whether you want to enable FilFinder plotting of filament spine

        Attributes
        ----------
        filspine : numpy.ndarray
           A 2D array of 1s and 0s defining the longest path through the filament mask
        """
        
        # Read beamwidth
        if isinstance(beamwidth, numbers.Number):
            if (self.header is not None):
                self.beamwidth = beamwidth * u.arcsec
            else:
                self.beamwidth = beamwidth * u.pix
        else:
            self.beamwidth = None
            raise TypeError("A beamwidth is required")

        # fil_finder
        ## Let fil_fineder deal with the beamwidth
        if (self.header is not None):
            fils = fil_finder_2D(self.image,
                                 header = self.header,
                                 beamwidth = self.beamwidth,
                                 distance = self.distance,
                                 mask = self.mask)
        ## scale-free
        else:
            fils = fil_finder_2D(self.image,
                                 beamwidth = self.beamwidth,
                                 skel_thresh = 15,
                                 mask = self.mask)
            ## 15 is chosen to be roughly 0.3 pc at the distance to Per B5 (260 pc).
            ## Consider allowing users to input in the future.

        # do the skeletonization
        fils.medskel(verbose=verbose)

        # Find shortest path through skeleton
        analysis = fils.analyze_skeletons(verbose=verbose)

        # Return the reults.
        self.filspine = fils.skeleton_longpath.astype(bool)
        if (self.header is not None):
            self.length = np.sum(analysis.lengths) * u.pc
            self.imgscale = fils.imgscale * u.pc
        else:
            self.length = np.sum(analysis.lengths) * u.pix
            self.imgscale = fils.imgscale * u.pix

        # Return a dictionary to store the key setup Parameters
        self._params['__init__']['imgscale'] = self.imgscale
        params = {'beamwidth': self.beamwidth}
        self._params['make_fil_spine'] = params

        # Return a dictionary to store the results
        self._results['make_fil_spine']['filspine'] = self.filspine
        self._results['make_fil_spine']['length'] = self.length

        return self

    def build_profile(self, pts_mask = None, samp_int=3, bins = None, shift = True, wrap = False, cut = True, cutdist = None):

        """
        Build the filament profile using the inputted or recently created filament spine

        Parameters
        ----------
        self: An instance of the radfil_class

        pts_mask: numpy.ndarray
            A 2D array masking out any regions from image array you don't want to sample; must be of boolean
            type and the same shape as the image array

        samp_int: integer (default=3)
            An integer indicating how frequently you'd like to make sample cuts
            across the filament.

        bins: int or 1D numpy.ndarray, optional (default=120)
            The number of bins or the actual bin edges you'd like to divide the whole profile (-cutdist to +cutdist) into, assuming nobins=False.
            If false, all of the individual profiles are binned by distance from r=0 pc and then the median column density
            in each of these bins is taken to determine the master profile

        norm_constant: float, optional (default=1e+22)
            Would you like to normalize your column densites (or flux) values by some normalization constant? If so
            enter it as a float or int

        shift: boolean (default = True)
            Indicates whether to shift the profile to center at the peak value.

        wrap: boolean (default = False)
            Indicates whether to wrap around the central pixel, so that the final profile
            will be a "half profile" with the peak near/at the center (depending on
            whether it's shifted).

        cut: boolean (default = True)
            Indicates whether to perform cuts when extracting the profile. Since
            the original spine found by `fil_finder_2D` is not likely differentiable
            everywhere, setting `cut = True` necessates a spline fit to smoothe
            the spine. See related documentation above.

            Setting `cut = False` will make `radfil` calculate a distance and a
            height/value for every pixel inside the mask.

        cutdist: float, optional (default = None)
            The distance out to which the profile is maked and stored in profile_masked.

        Attributes
        ----------


        xall, yall: 1D numpy.ndarray (list-like)
            All data points (with or without cutting).

        xbeforespline, ybeforespline: 1D numpy.ndarray (list-like)
            Positions of the "filament" identified by `fil_finder_2D`, in pixel
            units.  This is before smoothing done with `spline`.

        xspline, yspline: 1D numpy.ndarray (list-like)
            Positions of the spline points used for cuts, in pixel units.

        masterx, mastery: 1D numpy.ndarray (list-like)
            The profile (radial distances and height/column density/intensity)
            obtained by `profile_builder`.

        dictionary_cuts: Python dictionary
            A dictionary containing the profile (radian distances and height)
            for each cut along the spline, as two lists--one for the distance,
            and the other for the height.
        """


        # Read shift, wrap, cut, and samp_int
        ## shift
        if isinstance(shift, bool):
            self.shift = shift
        else:
            raise TypeError("shift has to be a boolean value. See documentation.")
        ## wrap
        if isinstance(wrap, bool):
            self.wrap = wrap
        else:
            raise TypeError("wrap has to be a boolean value. See documentation.")
        ## cut
        if isinstance(cut, bool):
            self.cutting = cut
        else:
            raise TypeError("cut has to be a boolean value. See documentation.")
        ## samp_int
        if isinstance(samp_int, int):
            self.samp_int = samp_int
        else:
            self.samp_int = None
            warnings.warn("samp_int has to be an integer; ignored for now. See documentation.")

        # Read the pts_mask and see if it needs padding
        if isinstance(pts_mask, np.ndarray) and (pts_mask.ndim == 2):
            if (self.padsize > 0):
                self.pts_mask = (np.pad(pts_mask,
                                        self.padsize,
                                        'constant',
                                        constant_values = 0)).astype(bool)
            else:
                self.pts_mask = pts_mask.astype(bool)
        else:
            self.pts_mask = None

        #extract x and y coordinates of filament spine
        pixcrd = np.where(self.filspine)

        # Sort these points by distance along the spine
        x, y = profile_tools.curveorder(pixcrd[1], pixcrd[0])
        self.xbeforespline, self.ybeforespline = x, y

        # If cut
        if self.cutting:
            # Filter out wrong samp_int
            if self.samp_int is None:
                raise TypeError("samp_int has to be an integer, when cut is True.")
            # Spline calculation:
            ##set the spline parameters
            k = 5
            nest = -1 # estimate of number of knots needed (-1 = maximal)
            ## find the knot points
            tckp, up, = splprep([x,y], k = k, nest = -1)
            ## evaluate spline
            xspline, yspline = splev(up, tckp)
            xprime, yprime = splev(up, tckp, der=1)
            ## Notice that the result containt points on the spline that are not
            ## evenly sampled.  This might introduce biases when using a single
            ## number `samp_int`.

            ## Plot the results
            fig=plt.figure(figsize=(8,8))
            ax=plt.gca()
            ax.imshow(self.mask, origin='lower', cmap='binary_r', interpolation='none')
            ax.plot(xspline, yspline, 'r', label='fit', lw=2, alpha=0.5)
            ax.set_xlim(int(np.min(np.where(self.mask==1)[1]*0.75)),int(np.max(np.where(self.mask==1)[1])*1.25))
            ax.set_ylim(int(np.min(np.where(self.mask==1)[0]*0.75)),int(np.max(np.where(self.mask==1)[0])*1.25))
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            self.fig, self.ax = fig, ax

            # Only points within pts_mask AND the original mask are used.
            if (self.pts_mask is not None):
                pts_mask = ((self.pts_mask[np.round(yspline[1:-1:self.samp_int]).astype(int),
                                           np.round(xspline[1:-1:self.samp_int]).astype(int)]) &\
                            (self.mask[np.round(yspline[1:-1:self.samp_int]).astype(int),
                                       np.round(xspline[1:-1:self.samp_int]).astype(int)]))
            else:
                pts_mask = (self.mask[np.round(yspline[1:-1:self.samp_int]).astype(int),
                                      np.round(xspline[1:-1:self.samp_int]).astype(int)])

            # Prepare for extracting the profiles
            self.xspline = xspline[1:-1:self.samp_int][pts_mask]
            self.yspline = yspline[1:-1:self.samp_int][pts_mask]
            self.points = np.asarray(zip(self.xspline, self.yspline))
            self.fprime = np.asarray(zip(xprime[1:-1:self.samp_int][pts_mask], yprime[1:-1:self.samp_int][pts_mask]))


            # Extract the profiles
            dictionary_cuts = defaultdict(list)
            if (self.imgscale.unit == u.pc):
                for n in range(len(self.points)):
                    profile = profile_tools.profile_builder(self, self.points[n], self.fprime[n], shift = self.shift, wrap = self.wrap)
                    cut_distance = profile[0]*self.imgscale.to(u.pc).value
                    dictionary_cuts['distance'].append(cut_distance) ## in pc
                    dictionary_cuts['profile'].append(profile[1])
                    dictionary_cuts['plot_peaks'].append(profile[2])
                    dictionary_cuts['plot_cuts'].append(profile[3])

                    ##
                    dictionary_cuts['mask_width'].append(geometry.LineString(profile[3]).length*self.imgscale.value)
                    ##
                    if isinstance(cutdist, numbers.Number):
                        dictionary_cuts['profile_masked'].append(np.ma.array(profile[1],\
                                                                             mask = (abs(cut_distance) >= cutdist)))
            elif (self.imgscale.unit == u.pix):
                for n in range(len(self.points)):
                    profile = profile_tools.profile_builder(self, self.points[n], self.fprime[n], shift = self.shift, wrap = self.wrap)
                    cut_distance = profile[0]*self.imgscale.to(u.pix).value  ## in pix
                    dictionary_cuts['distance'].append(cut_distance)
                    dictionary_cuts['profile'].append(profile[1])
                    dictionary_cuts['plot_peaks'].append(profile[2])
                    dictionary_cuts['plot_cuts'].append(profile[3])

                    ##
                    dictionary_cuts['mask_width'].append(geometry.LineString(profile[3]).length)
                    ##
                    if isinstance(cutdist, numbers.Number):
                        dictionary_cuts['profile_masked'].append(np.ma.array(profile[1],\
                                                                             mask = (abs(cut_distance) >= cutdist)))

            # Return the complete set of cuts. Including those outside `cutdist`.
            self.dictionary_cuts = dictionary_cuts
            ## Plot the peak positions if shift
            if self.shift:
                self.ax.plot(np.asarray(dictionary_cuts['plot_peaks'])[:, 0],
                             np.asarray(dictionary_cuts['plot_peaks'])[:, 1],
                             'b.', markersize = 10.)
        # if no cutting
        else:
            ## warnings.warn if samp_int exists.
            if (self.samp_int is not None):
                self.samp_int = None
                warnings.warn("samp_int is not used. cut is False.")
            ## warnings.warn if shift and/or wrap is True.
            if (self.shift or self.wrap):
                warnings.warn("shift and/or wrap are not used. cut is False.")
                self.shift, self.wrap = False, True

            # Only points within pts_mask AND the original mask are used.
            if (self.pts_mask is not None):
                pts_mask = ((self.pts_mask[np.round(self.ybeforespline).astype(int),
                                           np.round(self.xbeforespline).astype(int)]) &\
                            (self.mask[np.round(self.ybeforespline).astype(int),
                                       np.round(self.xbeforespline).astype(int)]))
            else:
                pts_mask = (self.mask[np.round(self.ybeforespline).astype(int),
                                      np.round(self.xbeforespline).astype(int)])

            # Make the line object with Shapely
            self.points = np.asarray(zip(self.xbeforespline[pts_mask], self.ybeforespline[pts_mask]))
            line = geometry.LineString(self.points)
            self.xspline, self.yspline, self.fprime = None, None, None

            ## Plot the results
            fig=plt.figure(figsize=(5,5))
            ax=plt.gca()
            ax.imshow(self.mask, origin='lower', cmap='binary_r', interpolation='none')
            ax.plot(line.xy[0], line.xy[1], 'r', label='fit', lw=2, alpha=0.25)
            ax.set_xlim(int(np.min(np.where(self.mask==1)[1]*0.75)),int(np.max(np.where(self.mask==1)[1])*1.25))
            ax.set_ylim(int(np.min(np.where(self.mask==1)[0]*0.75)),int(np.max(np.where(self.mask==1)[0])*1.25))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            self.fig, self.ax = fig, ax

            # Extract the distances and the heights
            dictionary_cuts = {}
            if (self.imgscale.unit == u.pc):
                dictionary_cuts['distance'] = [[line.distance(geometry.Point(coord))*self.imgscale.to(u.pc).value for coord in zip(np.where(self.mask)[1], np.where(self.mask)[0])]]
                dictionary_cuts['profile'] = [[self.image[coord[1], coord[0]] for coord in zip(np.where(self.mask)[1], np.where(self.mask)[0])]]
                dictionary_cuts['plot_peaks'] = None
                dictionary_cuts['plot_cuts'] = None

                ##
                if isinstance(cutdist, numbers.Number):
                    dictionary_cuts['profile_masked'] = np.ma.array(dictionary_cuts['profile'],
                                                                    mask = abs(np.asarray(dictionary_cuts['distance'])) >= cutdist)
            elif (self.imgscale.unit == u.pix):
                dictionary_cuts['distance'] = [[line.distance(geometry.Point(coord))*self.imgscale.to(u.pix).value for coord in zip(np.where(self.mask)[1], np.where(self.mask)[0])]]
                dictionary_cuts['profile'] = [[self.image[coord[1], coord[0]] for coord in zip(np.where(self.mask)[1], np.where(self.mask)[0])]]
                dictionary_cuts['plot_peaks'] = None
                dictionary_cuts['plot_cuts'] = None

                ##
                if isinstance(cutdist, numbers.Number):
                    dictionary_cuts['profile_masked'] = np.ma.array(dictionary_cuts['profile'],
                                                                    mask = abs(np.asarray(dictionary_cuts['distance'])) >= cutdist)
            self.dictionary_cuts = dictionary_cuts


        # Stack the result.
        ## xall, yall = np.concatenate(self.dictionary_cuts['distance']), np.concatenate(self.dictionary_cuts['profile'])
        mask_cutdist = np.concatenate([~singlecut.mask for singlecut in self.dictionary_cuts['profile_masked']])
        xall, yall = np.concatenate(self.dictionary_cuts['distance'])[mask_cutdist],\
                     np.concatenate(self.dictionary_cuts['profile'])[mask_cutdist]
        #xall, yall = xall[(xall >= (-self.cutdist/self.imgscale).decompose().value)&\
        #                   (xall < (self.cutdist/self.imgscale).decompose().value)],\
        #             yall[(xall >= (-self.cutdist/self.imgscale).decompose().value)&\
        #                  (xall < (self.cutdist/self.imgscale).decompose().value)]
        ## Store the values.
        self.xall = xall ## in pc
        self.yall = yall

        ### the following operations, including binning and fitting, should be done on self.xall and self.yall.


        # Bin the profiles (if nobins=False) or stack the profiles (if nobins=True)
        ## This step assumes linear binning.
        ## If the input is the number of bins:
        if isinstance(bins, numbers.Number) and (bins%1 == 0):
            self.binning = True
            bins = int(round(bins))
            minR, maxR = np.min(self.xall), np.max(self.xall)
            bins = np.linspace(minR, maxR, bins+1)
            masterx = bins[:-1]+.5*np.diff(bins)
            mastery = np.asarray([np.median(self.yall[((self.xall >= (X-.5*np.diff(bins)[0]))&\
                                  (self.xall < (X+.5*np.diff(bins)[0])))]) for X in masterx])

            self.bins = bins
        ## If the input is the edges of bins:
        elif isinstance(bins, np.ndarray) and (bins.ndim == 1):
            self.binning = True
            bins = bins
            masterx = bins[:-1]+.5*np.diff(bins) ## assumes linear binning.
            mastery = np.asarray([np.median(self.yall[((self.xall >= (X-.5*np.diff(bins)[0]))&\
                                  (self.xall < (X+.5*np.diff(bins)[0])))]) for X in masterx])

            self.bins = bins
        ## If the input is not bins-like.
        else:
            self.binning = False
            self.bins = None
            masterx = self.xall
            mastery = self.yall
            print "No binning is applied."

        # Close the figure
        #plt.close()

        # Return the profile sent to `fit_profile`.
        self.masterx=masterx
        self.mastery=mastery

        #return image, mask, and spine to original image dimensions without padding
        if self.padsize!=None and self.padsize!=0:
            self.image=self.image[self.padsize:self.image.shape[0]-self.padsize,self.padsize:self.image.shape[1]-self.padsize]
            self.mask=self.mask[self.padsize:self.mask.shape[0]-self.padsize,self.padsize:self.mask.shape[1]-self.padsize]
            self.filspine=self.filspine[self.padsize:self.filspine.shape[0]-self.padsize,self.padsize:self.filspine.shape[1]-self.padsize]

        # Return a dictionary to store the key setup Parameters
        self._params['__init__']['image'] = self.image ## (unpadded)
        self._params['__init__']['mask'] = self.mask ## This is the intersection between all the masks
        params = {'cutting': self.cutting,
                  'binning': self.binning,
                  'shift': self.shift,
                  'wrap': self.wrap,
                  'bins': self.bins,
                  'samp_int': self.samp_int,
                  'cutdist': cutdist}
        self._params['build_profile'] = params

        # Return a dictionary to store the results
        ## "points" are the spline points used for the cuts or
        ## the point collection of the original spine in the
        ## "no-cutting" case.
        ## "dictionary_cuts" are for plotting, mainly.
        results = {'points': self.points,
                   'xall': self.xall,
                   'yall': self.yall,
                   'masterx': self.masterx,
                   'mastery': self.mastery,
                   'dictionary_cuts': self.dictionary_cuts}
        self._results['build_profile'] = results

        return self

    def fit_profile(self, bgdist = None, fitdist = None, fitfunc=None, verbose=False, beamwidth=None, bgdegree = 1):

        """
        Fit a model to the filament's master profile

        Parameters
        ------
        self: An instance of the radfil_class

        fitdist: number-like
            The radial distance (in units of pc) out to which you'd like to fit your profile.

        bgdist: tuple-like, with a shape (2,)
            The radial distance range that defines the data points to be used in background subtraction.
            
        fitfunc: string
            Options include "Gaussian" or "Plummer"

        verbose: boolean,optional (default=False)
            Would you like to display the plots?

        bgdegree: integer (default = 1)
            The order of the polynomial used in background subtraction.  Active only when wrap = False.
            
        beamwidth: float or int
            If not inputed into the make_fil_spine method, beamwidth needs to be provided to calculate deconvolved FWHM of Gaussian/Plummer Fits
            If not provided, deconvolved FWHM values will be set to nan

        Attributes
        ------

        xbg, ybg: 1D numpy.ndarray (list-like)
            Data used for background subtraction.

        xfit, yfit: 1D numpy.ndarray (list-like)
            Data used in fitting.

        bgfit: astropy.modeling.functional_models (1st-order) or float (0th-order)
            The background removal information.

        profilefit: astropy.modeling.functional_models
            The fitting results.

        """
        
        #Check to make sure user entered valid function
        if fitfunc!="Plummer" and fitfunc!="Gaussian":
            raise ValueError("Reset fitfunc; You have not entered a valid function. Input 'Gaussian' or 'Plummer'")
            
        #Check whether beamwidth already exists, or whether they have inputed one here to compute deconvolved FWHM      
        if (hasattr(self,'beamwith')==False) & (type(beamwidth)!=None):
            if isinstance(beamwidth, numbers.Number):
                if (self.header is not None):
                    self.beamwidth = beamwidth * u.arcsec
                else:
                    self.beamwidth = beamwidth * u.pix
            else:
                self.beamwidth = None

        # Mask for bg removal
        ## take only bgdist, which should be a 2-tuple or 2-list
        if np.asarray(bgdist).shape == (2,):
            self.bgdist = np.sort(bgdist)
            ## below can be merged... ##########
            if self.wrap:
                maskbg = ((self.masterx >= self.bgdist[0])&\
                          (self.masterx < self.bgdist[1])&\
                          np.isfinite(self.mastery))
            else:
                maskbg = ((abs(self.masterx) >= self.bgdist[0])&\
                          (abs(self.masterx) < self.bgdist[1])&\
                          np.isfinite(self.mastery))

            if sum(maskbg) == 0.:
                raise ValueError("Reset bgdist; there is no data to fit for the background.")
        else:
            self.bgdist = None
            warnings.warn("No background removal will be performed.")

        # Mask for fitting
        ## Anything inside `fitdist` pc is used in fitting.
        if isinstance(fitdist, numbers.Number):
            self.fitdist = fitdist
            mask = ((self.masterx >= (-self.fitdist))&\
                    (self.masterx < self.fitdist)&\
                    np.isfinite(self.mastery))
            if sum(mask) == 0.:
                raise ValueError("Reset fitdist; there is no data inside fitdist.")
        ## Fit all data if no fitdist
        else:
            self.fitdist = None
            ## Just fool-proof
            mask = (np.isfinite(self.masterx)&\
                    np.isfinite(self.mastery))
            if sum(mask) == 0.:
                raise ValueError("Reset fitdist; there is no data inside fitdist.")

        # Fit for the background, and remove
        ## If bgdist (yes, background removal.)
        if np.asarray(self.bgdist).shape == (2,):
            ## In the case where the profile is wrapped, simply take the mean in the background.
            ## This is because that a linear fit (with a slope) with only one side is not definite.
            if self.wrap:
                xbg, ybg = self.masterx, self.mastery
                xbg, ybg = xbg[maskbg], ybg[maskbg]
                self.xbg, self.ybg = xbg, ybg
                self.bgfit = np.median(self.ybg) ### No fitting!
                self.ybg_filtered = None ## no filtering during background removal
                ## Remove bg without fitting (or essentially a constant fit).
                xfit, yfit = self.masterx[mask], self.mastery[mask]
                yfit = yfit - self.bgfit
                print "The profile is wrapped. Use the 0th order polynomial in BG subtraction."
            ## A first-order bg removal is carried out only when the profile is not wrapped.
            else:
                ## Fit bg
                xbg, ybg = self.masterx, self.mastery
                xbg, ybg = xbg[maskbg], ybg[maskbg]
                self.xbg, self.ybg = xbg, ybg
                bg_init = models.Polynomial1D(degree = bgdegree) ##########
                fit_bg = fitting.LinearLSQFitter()
                
                ## outlier removal; use sigma clipping, set to 3 sigmas
                fit_bg_or = fitting.FittingWithOutlierRemoval(fit_bg, sigma_clip,
                                                              niter=10, sigma=3.)
                bg = fit_bg(bg_init, self.xbg, self.ybg)
                data_or, bg_or = fit_bg_or(bg_init, self.xbg, self.ybg)
                self.bgfit = bg_or.copy()
                self.ybg_filtered = data_or ## a masked array returned by the outlier removal
                
                ## Remove bg and prepare for fitting
                xfit, yfit = self.masterx[mask], self.mastery[mask]
                yfit = yfit - self.bgfit(xfit)
                
        ## If no bgdist
        else:
            self.bgfit = None
            self.xbg, self.ybg = None, None
            self.ybg_filtered = None
            ## Set up fitting without bg removal.
            xfit, yfit = self.masterx[mask], self.mastery[mask]
        self.xfit, self.yfit = xfit, yfit

        # Fit Model
        ## Gaussian model
        if fitfunc=="Gaussian":
            g_init = models.Gaussian1D(amplitude = .8*np.max(self.yfit),
                                    mean = 0.,
                                    stddev=np.std(self.xfit),
                                    fixed = {'mean': True},
                                    bounds = {'amplitude': (0., np.inf),
                                             'stddev': (0., np.inf)})
            fit_g = fitting.LevMarLSQFitter()
            g = fit_g(g_init, self.xfit, self.yfit)
            self.profilefit = g.copy()
            print '==== Gaussian ===='
            print 'amplitude: %.3E'%self.profilefit.parameters[0]
            print 'width: %.3f'%self.profilefit.parameters[2]
            
        elif fitfunc=="Plummer":
            g_init = Plummer1D(amplitude = .8*np.max(self.yfit),
                            powerIndex=2.,
                            flatteningRadius = np.std(self.xfit))
                            
            fit_g = fitting.LevMarLSQFitter()
            g = fit_g(g_init, self.xfit, self.yfit)
            self.profilefit = g.copy()
            self.profilefit.parameters[2] = abs(self.profilefit.parameters[2]) #Make sure R_flat always positive
            print '==== Plummer-like ===='
            print 'amplitude: %.3E'%self.profilefit.parameters[0]
            print 'p: %.3f'%self.profilefit.parameters[1]
            print 'R_flat: %.3f'%self.profilefit.parameters[2]
            
        else:
            raise ValueError("Reset fitfunc; no valid function entered. Options include 'Gaussian' or 'Plummer'")

            
        ### Plot background fit if bgdist is not none ###
        if bgdist!=None:
            fig, ax = plt.subplots(figsize = (8, 8.), ncols = 1, nrows = 2)
            axis = ax[0]
            axis.plot(self.xall, self.yall, 'k.', markersize = 8., alpha = .1)
            [axis.axvline(_bglineplot, linewidth=1, color='g', ls='dashed') for _bglineplot in [self.bgdist[0],self.bgdist[1],-1*self.bgdist[0],-1*self.bgdist[1]]]
            axis.plot(np.linspace(np.min(self.xall),np.max(self.xall),100), np.linspace(np.min(self.xall),np.max(self.xall),100)*self.bgfit.parameters[1]+self.bgfit.parameters[0],'g-', lw=3)
            axis.set_xticklabels([])
            axis.tick_params(labelsize=14)
            
            xplot = self.xall
            yplot = self.yall - self.bgfit(xplot)
            
            #Mark BG subtraction boundaries#
            axis.add_patch(Rectangle((self.bgdist[0],axis.get_ylim()[0]),np.abs(self.bgdist[1]-self.bgdist[0]), axis.get_ylim()[1]-axis.get_ylim()[0], facecolor="green",alpha=0.02))
            axis.add_patch(Rectangle((-1*self.bgdist[1],axis.get_ylim()[0]),np.abs(self.bgdist[1]-self.bgdist[0]), axis.get_ylim()[1]-axis.get_ylim()[0], facecolor="green",alpha=0.02))
                       
            #Add labels#
            axis.text(0.03, 0.95,"m={:.2e}\nb={:.2e}".format(self.bgfit.parameters[1],self.bgfit.parameters[0]),ha='left',va='top', fontsize=14, fontweight='bold',transform=axis.transAxes,bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':1.0, 'pad':1})
            axis.text(0.85, 0.95,"Background\nFitting", ha='center',va='top', fontsize=20, fontweight='bold',transform=axis.transAxes,bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':1.0, 'pad':1})
                        
            #Adjust axes limits
            axis.set_xlim(np.min(self.xall), np.max(self.xall))
            axis.set_ylim(np.percentile(self.yall,0)-np.abs(0.5*np.percentile(self.yall,0)),np.percentile(self.yall,99.9)+np.abs(0.25*np.percentile(self.yall,99.9)))
            
            axis=ax[1]

        else:
            fig, ax = plt.subplots(figsize = (8, 4.), ncols = 1, nrows = 1)
            axis = ax
            
            xplot=self.xall
            yplot=self.yall
            

        ## Plot model 
        axis.plot(xplot, yplot, 'k.', markersize = 8., alpha = .1)
        [axis.axvline(_fitlineplot, linewidth=1, color='b', ls='dashed') for _fitlineplot in [-1*self.fitdist,self.fitdist]]
        axis.plot(np.linspace(np.min(xplot),np.max(xplot),100), self.profilefit(np.linspace(np.min(xplot),np.max(xplot),100)), 'b-', lw = 3., alpha = .6)
        axis.add_patch(Rectangle((-1*self.fitdist, axis.get_ylim()[0]),self.fitdist*2, axis.get_ylim()[1]-axis.get_ylim()[0], facecolor="b", alpha=0.02))
        
        #Adjust axis limit based on percentiles of data
        axis.set_xlim(np.min(self.xall), np.max(self.xall))
        axis.set_ylim(np.percentile(yplot,0)-np.abs(0.5*np.percentile(yplot,0)),np.percentile(yplot,99.9)+np.abs(0.25*np.percentile(yplot,99.9)))
        
        axis.text(0.03, 0.95,"{}={:.2e}\n{}={:.2f}\n{}={:.2f}".format(self.profilefit.param_names[0],self.profilefit.parameters[0],self.profilefit.param_names[1],self.profilefit.parameters[1],self.profilefit.param_names[2],self.profilefit.parameters[2]),ha='left',va='top', fontsize=14, fontweight='bold',transform=axis.transAxes,bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':1.0, 'pad':1})
        axis.text(0.85, 0.95,"{}\nFitting".format(fitfunc), ha='center',va='top', fontsize=20, fontweight='bold',transform=axis.transAxes,bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':1.0, 'pad':1})
        axis.tick_params(labelsize=14)

        #add axis info
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        fig.text(0.5, -0.05, "Radial Distance ({})".format(str(self.imgscale.unit)),fontsize=25,ha='center')
        fig.text(-0.05, 0.5, "Profile Height",fontsize=25,va='center',rotation=90)

        # Return a dictionary to store the key setup Parameters
        params = {'bgdist': self.bgdist,
                  'fitdist': self.fitdist}
        self._params['fit_profile'] = params

        # Return a dictionary to store the results
        ## All the fits are `astropy.model` objects.
        results = {'bgfit': self.bgfit,
                   'profilefit': self.profilefit,
                   'xbg': self.xbg,
                   'ybg': self.ybg,
                   'xfit': self.xfit,
                   'yfit': self.yfit}
        self._results['fit_profile'] = results
                           
        if fitfunc=="Gaussian":
            FWHM= 2.*np.sqrt(2.*np.log(2.))*self.profilefit.parameters[2]
            
            if self.beamwidth!=None:
                 
                if (self.beamwidth.unit == u.arcsec) and (self.imgscale_ang is not None):
                    beamwidth_phys = (self.beamwidth/self.imgscale_ang).decompose()*self.imgscale.value
                    print 'Physical Size of the Beam:', beamwidth_phys*self.imgscale.unit
                
                    if np.isfinite(np.sqrt(FWHM**2.-beamwidth_phys**2.)):
                        FWHM_deconv = np.sqrt(FWHM**2.-beamwidth_phys**2.).value
                    else:
                        FWHM_deconv = np.nan
                        warnings.warn("The Gaussian width is not resolved.")
                
                elif (self.beamwidth.unit == u.pix):
                    beamwidth_phys = self.beamwidth.value
                    print 'Beamwidth in the Pixel Unit:', self.beamwidth
                
                    if np.isfinite(np.sqrt(FWHM**2.-beamwidth_phys**2.)):
                        FWHM_deconv = np.sqrt(FWHM**2.-beamwidth_phys**2.).value
                    else:
                        FWHM_deconv = np.nan
                        warnings.warn("The width is not resolved.")
                else:
                    FWHM_deconv = np.nan
                    warnings.warn("A beamwidth is not found. Deconvolved FWHMs cannot be derived.")
                
            else:
                    FWHM_deconv = np.nan
                    warnings.warn("A beamwidth is not found. Deconvolved FWHMs cannot be derived.")
                
                
        if fitfunc=="Plummer":
        
            FWHM=2.*self.profilefit.parameters[2]*np.sqrt(2.**(2./(self.profilefit.parameters[1]-1.)) - 1.)

            if self.beamwidth!=None:
                if (self.beamwidth.unit == u.arcsec) and (self.imgscale_ang is not None):
                    beamwidth_phys = (self.beamwidth/self.imgscale_ang).decompose()*self.imgscale.value
                    print 'Physical Size of the Beam:', beamwidth_phys*self.imgscale.unit
                
                    if np.isfinite(np.sqrt(FWHM**2.-beamwidth_phys**2.)):
                        FWHM_deconv = np.sqrt(FWHM**2.-beamwidth_phys**2.).value
                    else:
                        FWHM_deconv = np.nan
                        warnings.warn("The width is not resolved.")
                
                elif (self.beamwidth.unit == u.pix):
                    beamwidth_phys = self.beamwidth.value
                    print 'Beamwidth in the Pixel Unit:', self.beamwidth
                
                    if np.isfinite(np.sqrt(FWHM**2.-beamwidth_phys**2.)):
                        FWHM_deconv = np.sqrt(FWHM**2.-beamwidth_phys**2.).value
                    else:
                        FWHM_deconv = np.nan
                        warnings.warn("The width is not resolved.")
                else:
                    FWHM_deconv = np.nan
                    warnings.warn("A beamwidth is not found. Deconvolved FWHMs cannot be derived.")     
            else:
                FWHM_deconv = np.nan
                warnings.warn("A beamwidth is not found. Deconvolved FWHMs cannot be derived.")  
                        
        self._results['FWHM'] = FWHM
        self._results['FWHM_deconv'] = FWHM_deconv

        ###########################
        #OLD CODE RECORDS (Can delete later)
        """
        # Return FWHM from both Plummer and Guassian
        FWHM = {'gaussian': 2.*np.sqrt(2.*np.log(2.))*self.profilefit_gaussian.parameters[2],
                'plummer': 2.*self.profilefit_plummer.parameters[2]*\
                           np.sqrt(2.**(2./(self.profilefit_plummer.parameters[1]-1.)) - 1.)}
        if (self.beamwidth.unit == u.arcsec) and (self.imgscale_ang is not None):
            beamwidth_phys = (self.beamwidth/self.imgscale_ang).decompose()*self.imgscale.value
            print 'Physical Size of the Beam:', beamwidth_phys*self.imgscale.unit
            if np.isfinite(np.sqrt(FWHM['gaussian']**2.-beamwidth_phys**2.)):
                FWHM['gaussian_deconvolved'] = np.sqrt(FWHM['gaussian']**2.-beamwidth_phys**2.).value
            else:
                FWHM['gaussian_deconvolved'] = np.nan
                warnings.warn("The Gaussian width is not resolved.")
            if np.isfinite(np.sqrt(FWHM['plummer']**2.-beamwidth_phys**2.)):
                FWHM['plummer_deconvolved'] = np.sqrt(FWHM['plummer']**2.-beamwidth_phys**2.).value
            else:
                FWHM['plummer_deconvolved'] = np.nan
                warnings.warn("The Plummer width is not resolved.")
                
                
        elif (self.beamwidth.unit == u.pix):
            beamwidth_phys = self.beamwidth.value
            print 'Beamwidth in the Pixel Unit:', self.beamwidth
            if np.isfinite(np.sqrt(FWHM['gaussian']**2.-beamwidth_phys**2.)):
                FWHM['gaussian_deconvolved'] = np.sqrt(FWHM['gaussian']**2.-beamwidth_phys**2.).value
            else:
                FWHM['gaussian_deconvolved'] = np.nan
                warnings.warn("The Gaussian width is not resolved.")
            if np.isfinite(np.sqrt(FWHM['plummer']**2.-beamwidth_phys**2.)):
                FWHM['plummer_deconvolved'] = np.sqrt(FWHM['plummer']**2.-beamwidth_phys**2.).value
            else:
                FWHM['plummer_deconvolved'] = np.nan
                warnings.warn("The Plummer width is not resolved.")
        else:
            FWHM['gaussian_deconvolved'] = np.nan
            FWHM['plummer_deconvolved'] = np.nan
            warnings.warn("A beamwidth is not found. Deconvolved FWHMs cannot be derived.")
            
        """    

        return self


    def plotter(self):
        '''
        Return a `radfil.plot.RadFilPlotter` class.
        '''
        print "This is a feature under active development. Use at your own risk."

        from radfil import plot
        return plot.RadFilPlotter(self)
