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

import astropy.units as u
import astropy.constants as c
from astropy.modeling import models, fitting
from astropy.io import fits
from fil_finder import fil_finder_2D
import shapely.geometry as geometry

from radfil import profile_tools
from plummer import Plummer1D



#bgoptions={'flat':False,'sloping':True}


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

    def __init__(self, image, mask, header, distance, filspine=None, padsize=None, imgscale=None):

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
        else:
            raise TypeError("The imput `header` has to be a fits header.")

        # Read distance
        ## `self.distance` is in pc.
        if isinstance(distance, numbers.Number):
            self.distance = float(distance) * u.pc
        else:
            raise TypeError("The input `distance` has to be a float number or an integer.")


        # Read filspine/define filspine
        if (isinstance(filspine, np.ndarray) and (filspine.ndim == 2)) or (filspine is None):
            self.filspine = filspine

            # calculate "imgscale" when the spine is provided. This saves users
            # the trouble to run `make_fil_spine`.
            if (isinstance(filspine, np.ndarray) and (filspine.ndim == 2)):
                # Calculate pixel scale ("imgscale"), in the unit of pc/Deal with non-standard fits header
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
                        raise TypeError("Please specify a proper `imgscale` in parsec if the information is not in the header.")

        else:
            self.filspine = None
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


    def make_fil_spine(self,beamwidth=None,verbose=False):

        """
        Create filament spine using the FilFinder package 'shortest path' option

        Parameters:
         ----------
        beamwidth: float
            A float in units of arcseconds indicating the beamwidth of the image
            array.  When corresponding keys are in the header file, the header
            will be used to determine the beamwidth.  Only when the header does
            not containt information regarding the beamwidth, is the input
            `beamwidth` used in the calculation.

        verbose: boolean
            A boolean indicating whether you want to enable FilFinder plotting of filament spine

        Attributes
        ----------
        filspine : numpy.ndarray
           A 2D array of 1s and 0s defining the longest path through the filament mask
        """

        # Read beamwidth
        if isinstance(beamwidth, numbers.Number):
            self.beamwidth = beamwidth * u.arcsec
        else:
            self.beamwidth = None
            warnings.warn("A beamwidth is needed if the header does not contain the beam information.")

        # fil_finder
        ## Let fil_fineder deal with the beamwidth
        fils = fil_finder_2D(self.image,
                             self.header,
                             beamwidth=self.beamwidth,
                             distance=self.distance,
                             mask=self.mask)

        # do the skeletonization
        fils.medskel(verbose=verbose)

        # Find shortest path through skeleton
        analysis = fils.analyze_skeletons(verbose=verbose)

        # Return the reults.
        self.filspine = fils.skeleton_longpath.astype(bool)
        self.length = np.sum(analysis.lengths) * u.pc
        self.imgscale = fils.imgscale * u.pc

        return self

    def build_profile(self, pts_mask = None, samp_int=3, bins = None, shift = True, wrap = False, cut = True):

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

        distpc: numpy.ndarray
            A numpy array containing the physical size of each "local width" cut. Can take
            the mean/median to determine representative width of the filament mask
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
            k = 5 # spline order ## why 5 when scipy suggested 3?
            nest = -1 # estimate of number of knots needed (-1 = maximal)
            ## find the knot points
            tckp, up, = splprep([x,y], k = k, nest = -1)
            ## evaluate spline
            xspline, yspline = splev(up, tckp)
            xprime, yprime = splev(up, tckp, der=1)
            ## Notice that the result containt points on the spline that are not
            ## evenly sampled.  This might introduce biase when using a single
            ## number `samp_int`.

            ## Plot the results
            fig=plt.figure(figsize=(5,5))
            ax=plt.gca()
            ax.imshow(self.mask, origin='lower', cmap='binary_r', interpolation='none')
            ax.plot(xspline, yspline, 'r', label='fit', lw=2, alpha=0.25)
            ax.set_xlim(-.5, self.mask.shape[1]-.5)
            ax.set_ylim(-.5, self.mask.shape[0]-.5)
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
            for n in range(len(self.points)):
                profile = profile_tools.profile_builder(self, self.points[n], self.fprime[n], shift = self.shift, wrap = self.wrap)
                dictionary_cuts['distance'].append(profile[0]*self.imgscale.to(u.pc).value) ## in pc
                dictionary_cuts['profile'].append(profile[1])
                dictionary_cuts['plot_peaks'].append(profile[2])
                dictionary_cuts['plot_cuts'].append(profile[3])

            # Return the complete set of cuts. Including those outside `cutdist`.
            self.dictionary_cuts = dictionary_cuts
            ## Plot the peak positions if shift
            if self.shift:
                self.ax.plot(np.asarray(dictionary_cuts['plot_peaks'])[:, 0],
                             np.asarray(dictionary_cuts['plot_peaks'])[:, 1],
                             'b.', markersize = 6.)
        # if no cutting
        else:
            ## warnings.warn if samp_int exists.
            if (self.samp_int is not None):
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
            ax.set_xlim(-.5, self.mask.shape[1]-.5)
            ax.set_ylim(-.5, self.mask.shape[0]-.5)
            self.fig, self.ax = fig, ax

            # Extract the distances and the heights
            dictionary_cuts = {}
            dictionary_cuts['distance'] = [[line.distance(geometry.Point(coord))*self.imgscale.to(u.pc).value for coord in zip(np.where(self.mask)[1], np.where(self.mask)[0])]]
            dictionary_cuts['profile'] = [[self.image[coord[1], coord[0]] for coord in zip(np.where(self.mask)[1], np.where(self.mask)[0])]]
            dictionary_cuts['plot_peaks'] = None
            dictionary_cuts['plot_cuts'] = None
            self.dictionary_cuts = dictionary_cuts



        # Stack the result.
        xall, yall = np.concatenate(self.dictionary_cuts['distance']), np.concatenate(self.dictionary_cuts['profile'])
        #xall, yall = xall[(xall >= (-self.cutdist/self.imgscale).decompose().value)&\
        #                   (xall < (self.cutdist/self.imgscale).decompose().value)],\
        #             yall[(xall >= (-self.cutdist/self.imgscale).decompose().value)&\
        #                  (xall < (self.cutdist/self.imgscale).decompose().value)]
        ## Store the values.
        self.xall = xall ## in pc
        self.yall = yall


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
        ## If the input is the edges of bins:
        elif isinstance(bins, np.ndarray) and (bins.ndim == 1):
            self.binning = True
            bins = bins
            masterx = bins[:-1]+.5*np.diff(bins) ## assumes linear binning.
            mastery = np.asarray([np.median(self.yall[((self.xall >= (X-.5*np.diff(bins)[0]))&\
                                  (self.xall < (X+.5*np.diff(bins)[0])))]) for X in masterx])
        ## If the input is not bins-like.
        else:
            self.binning = False
            masterx = self.xall
            mastery = self.yall
            print "No binning is applied."




        # Return the profile sent to `fit_profile`.
        self.masterx=masterx
        self.mastery=mastery

        #return image, mask, and spine to original image dimensions without padding
        if self.padsize!=None and self.padsize!=0:
            self.image=self.image[self.padsize:self.image.shape[0]-self.padsize,self.padsize:self.image.shape[1]-self.padsize]
            self.mask=self.mask[self.padsize:self.mask.shape[0]-self.padsize,self.padsize:self.mask.shape[1]-self.padsize]
            self.filspine=self.filspine[self.padsize:self.filspine.shape[0]-self.padsize,self.padsize:self.filspine.shape[1]-self.padsize]

        return self

    def fit_profile(self, bgdist = .05, bgdistfrom = 'outside', fitdist = 3., verbose=False):

        """
        Fit a model to the filament's master profile

        Parameters
        ------
        self: An instance of the radfil_class

        fitdist: number-like
            The radial distance (in units of pc) out to which you'd like to fit your profile.

        bgdist: number-like
            The distance in pc for the background removal.  See bgdistfrom.

        bgdistfrom: 'outside' or 'inside'
            This indicates whether bgdist is from "outside" or "inside".  For example,
                bgdist = .1, bgdistfrom = 'outside'
            This means that data in the outtermost 0.1 pc will be used for background removal.

            Another example,
                bgdist = .5, bgdistfrom = 'inside'
            This means that data 0.5 pc or further away from the spine will be used for background removal.

        verbose: boolean,optional (default=False)
            Would you like to display the plots?

        Attributes
        ------

        xbg, ybg: 1D numpy.ndarray (list-like)
            Data used for background subtraction.

        xfit, yfit: 1D numpy.ndarray (list-like)
            Data used in fitting.

        bgfit: astropy.modeling.functional_models (1st-order) or float (0th-order)
            The background removal information.

        profilefit_gaussian, profilefit_plummer: astropy.modeling.functional_models
            The fitting results.


        """
        # Mask for bg removal
        ## The outter most `bgdist` pc is treated as the background
        if isinstance(bgdist, numbers.Number) and (bgdistfrom.lower() == 'outside'):
            self.bgdist = bgdist ## assuming pc
            self.bgdistfrom = bgdistfrom.lower()
            if self.wrap:
                maskbg = ((self.masterx >= (np.max(self.masterx)-bgdist))&\
                        np.isfinite(self.mastery))
            else:
                maskbg = (((self.masterx < (np.min(self.masterx)+bgdist))|\
                        (self.masterx >= (np.max(self.masterx)-bgdist)))&\
                        np.isfinite(self.mastery))

            if sum(maskbg) == 0.:
                raise ValueError("Reset bgdist; there is no data to fit for the background.")
        ## Anything outside `bgdist` pc is treated as the background.
        elif isinstance(bgdist, numbers.Number) and (bgdistfrom.lower() == 'inside'):
            self.bgdist = bgdist ## assuming pc
            self.bgdistfrom = bgdistfrom.lower()
            maskbg = (((self.masterx < (-bgdist))|\
                    (self.masterx >= bgdist))&\
                    np.isfinite(self.mastery))

            if sum(maskbg) == 0.:
                raise ValueError("Reset bgdist; there is no data to fit for the background.")
        ## No background removal.
        else:
            self.bgdist = None
            self.bgdistfrom = None
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
        if isinstance(self.bgdist, numbers.Number):
            ## In the case where the profile is wrapped, simply take the mean in the background.
            ## This is because that a linear fit (with a slope) with only one side is not definite.
            if self.wrap:
                xbg, ybg = self.masterx, self.mastery
                xbg, ybg = xbg[maskbg], ybg[maskbg]
                self.xbg, self.ybg = xbg, ybg
                self.bgfit = np.median(self.ybg) ### No fitting!
                ## Remove bg without fitting (or essentially a constant fit).
                xfit, yfit = self.masterx[mask], self.mastery[mask]
                yfit = yfit - self.bgfit
            ## A first-order bg removal is carried out only when the profile is not wrapped.
            else:
                ## Fit bg
                xbg, ybg = self.masterx, self.mastery
                xbg, ybg = xbg[maskbg], ybg[maskbg]
                self.xbg, self.ybg = xbg, ybg
                bg_init = models.Linear1D(intercept = np.mean(self.ybg))
                fit_bg = fitting.LinearLSQFitter()
                bg = fit_bg(bg_init, self.xbg, self.ybg)
                self.bgfit = bg.copy()
                ## Remove bg and prepare for fitting
                xfit, yfit = self.masterx[mask], self.mastery[mask]
                yfit = yfit - self.bgfit(xfit)
        ## If no bgdist
        else:
            self.bgfit = None
            self.xbg, self.ybg = None, None
            ## Set up fitting without bg removal.
            xfit, yfit = self.masterx[mask], self.mastery[mask]
        self.xfit, self.yfit = xfit, yfit



        # Fit (both) models
        ## Gaussian model
        g_init = models.Gaussian1D(amplitude = .8*np.max(self.yfit),
                                   mean = 0.,
                                   stddev=np.std(self.xfit),
                                   fixed = {'mean': True},
                                   bounds = {'amplitude': (0., np.inf),
                                             'stddev': (0., np.inf)})
        fit_g = fitting.LevMarLSQFitter()
        g = fit_g(g_init, self.xfit, self.yfit)
        self.profilefit_gaussian = g.copy()
        print '==== Gaussian ===='
        print 'amplitude: %.3E'%self.profilefit_gaussian.parameters[0]
        print 'width: %.3f'%self.profilefit_gaussian.parameters[2]
        ## Plummer model
        g_init = Plummer1D(amplitude = .8*np.max(self.yfit),
                           powerIndex=2.,
                           flatteningRadius = np.std(self.xfit),
                           bounds = {'amplitude': (0., np.inf),
                                     'powerIndex': (1., np.inf),
                                     'flatteningRadius': (0., np.inf)})
        fit_g = fitting.LevMarLSQFitter()
        g = fit_g(g_init, self.xfit, self.yfit)
        self.profilefit_plummer = g.copy()
        print '==== Plummer-like ===='
        print 'amplitude: %.3E'%self.profilefit_plummer.parameters[0]
        print 'p: %.3f'%self.profilefit_plummer.parameters[1]
        print 'R_flat: %.3f'%self.profilefit_plummer.parameters[2]


        # Plot results #########################################################
        fig, ax = plt.subplots(figsize = (7., 5.), ncols = 2, nrows = 2)
        ## plummer(+bgfit, if bgfit)
        axis = ax[0, 0]
        axis.plot(self.xall, self.yall, 'k.', markersize = 8., alpha = .05)
        ## Plot bins if binned.
        if self.binning:
            stepx, stepy = np.zeros(len(self.masterx)*2)*np.nan, np.zeros(len(self.mastery)*2) * np.nan
            stepx[::2] = self.masterx-.5*np.diff(self.masterx)[0]  ## assuming linear binning
            stepx[1::2] = self.masterx+.5*np.diff(self.masterx)[0]
            stepy[::2], stepy[1::2] = self.mastery, self.mastery
            axis.plot(stepx, stepy, 'k-', alpha = .4)
        ## Plot bg if bg removed.
        if isinstance(self.bgdist, numbers.Number):
            if self.wrap:
                axis.plot(self.xbg, self.ybg, 'g.', markersize = 8., alpha = .15)
                axis.plot(self.xfit, self.yfit+self.bgfit, 'b.', markersize = 8., alpha = .15)
                ## Plot the fits (profilefit + bgfit)
                xplot = np.linspace(np.min(self.xall), np.max(self.xall), 100)
                axis.plot(xplot, self.bgfit+self.profilefit_plummer(xplot), 'b-', lw = 3., alpha = .6)
            else:
                axis.plot(self.xbg, self.ybg, 'g.', markersize = 8., alpha = .15)
                axis.plot(self.xfit, self.yfit+self.bgfit(self.xfit), 'b.', markersize = 8., alpha = .15)
                ## Plot the fits (profilefit + bgfit)
                xplot = np.linspace(np.min(self.xall), np.max(self.xall), 100)
                axis.plot(xplot, self.bgfit(xplot)+self.profilefit_plummer(xplot), 'b-', lw = 3., alpha = .6)


        else:
            axis.plot(self.xfit, self.yfit, 'b.', markersize = 8., alpha = .15)
            ## Plot the fits (profilefit)
            xplot = np.linspace(np.min(self.xall), np.max(self.xall), 100)
            axis.plot(xplot, self.profilefit_plummer(xplot), 'b-', lw = 3., alpha = .6)
        ## Adjust the plot
        axis.set_xlim(np.min(self.xall), np.max(self.xall))
        #axis.set_yscale('log')
        axis.set_xticklabels([])

        ## plummer residual
        axis = ax[0, 1]
        axis.plot(self.xfit, self.yfit-self.profilefit_plummer(self.xfit), 'b.', markersize = 8., alpha = .3)
        ## Adjust the plot
        axis.set_xlim(np.min(self.xall), np.max(self.xall))
        axis.set_xticklabels([])
        axis.set_yticklabels([])

        ## gaussian(+bgfit, if bgfit)
        axis = ax[1, 0]
        axis.plot(self.xall, self.yall, 'k.', markersize = 8., alpha = .15)
        ## Plot bins if binned.
        if self.binning:
            stepx, stepy = np.zeros(len(self.masterx)*2)*np.nan, np.zeros(len(self.mastery)*2) * np.nan
            stepx[::2] = self.masterx-.5*np.diff(self.masterx)[0]  ## assuming linear binning
            stepx[1::2] = self.masterx+.5*np.diff(self.masterx)[0]
            stepy[::2], stepy[1::2] = self.mastery, self.mastery
            axis.plot(stepx, stepy, 'k-', alpha = .4)
        ## Plot bg if bg removed.
        if isinstance(self.bgdist, numbers.Number):
            if self.wrap:
                axis.plot(self.xbg, self.ybg, 'g.', markersize = 8., alpha = .15)
                axis.plot(self.xfit, self.yfit+self.bgfit, 'r.', markersize = 8., alpha = .15)
                ## Plot the fits (profilefit + bgfit)
                xplot = np.linspace(np.min(self.xall), np.max(self.xall), 100)
                axis.plot(xplot, self.bgfit+self.profilefit_gaussian(xplot), 'r-', lw = 3., alpha = .6)
            else:
                axis.plot(self.xbg, self.ybg, 'g.', markersize = 8., alpha = .15)
                axis.plot(self.xfit, self.yfit+self.bgfit(self.xfit), 'r.', markersize = 8., alpha = .15)
                ## Plot the fits (profilefit + bgfit)
                xplot = np.linspace(np.min(self.xall), np.max(self.xall), 100)
                axis.plot(xplot, self.bgfit(xplot)+self.profilefit_gaussian(xplot), 'r-', lw = 3., alpha = .6)

        else:
            axis.plot(self.xfit, self.yfit, 'r.', markersize = 8., alpha = .15)
            ## Plot the fits (profilefit)
            xplot = np.linspace(np.min(self.xall), np.max(self.xall), 100)
            axis.plot(xplot, self.profilefit_gaussian(xplot), 'r-', lw = 3., alpha = .6)
        ## Adjust the plot
        axis.set_xlim(np.min(self.xall), np.max(self.xall))
        #axis.set_yscale('log')

        ## gaussian residual
        axis = ax[1, 1]
        axis.plot(self.xfit, self.yfit-self.profilefit_gaussian(self.xfit), 'r.', markersize = 8., alpha = .3)
        ## Adjust the plot
        axis.set_xlim(np.min(self.xall), np.max(self.xall))
        axis.set_yticklabels([])




        return self


    def plotter(self):
        '''
        Return a `radfil.plot.RadFilPlotter` class.
        '''

        from radfil import plot
        return plot.RadFilPlotter(self)
