import numpy as np
from collections import defaultdict
import types
from scipy.interpolate import splprep
from scipy.interpolate import splev
import matplotlib.pyplot as plt
import lmfit
from lmfit.models import Model
from lmfit import Parameters
import sys
from radfil import profile_tools
import astropy.units as u
import astropy.constants as c
from astropy.io import fits
import math
from matplotlib.colors import LogNorm
from scipy.optimize import least_squares
import numbers
from fil_finder import fil_finder_2D


# Plummer-like function for profile fitting.
def plummer(r,N_0,R_flat,p):
    return (N_0)/(1+(r/R_flat)**2)**((p-1)/2.0)

# Gaussian for profile fitting.
def gaussian(r, amp, wid):
    return (amp) * np.exp(-1 * np.power(r, 2) / (2 * np.power(wid, 2)))

bgoptions={'flat':False,'sloping':True}


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
                        raise Warning("The keyword `imgscale`, instead of the header, is used in calculations of physical distances.")
                    else:
                        raise TypeError("Please specify a proper `imgscale` in parsec if the information is not in the header.")

        else:
            self.filspine = None
            raise Warning("The input `filspine` has to be a 2D numpy array. Ignore for now.")




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
            raise Warning("A beamwidth is needed if the header does not contain the beam information.")

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

    def build_profile(self,cutdist=3.0,pts_mask=None,samp_int=3,bins=120,save_mask=None,nobins=True,norm_constant=1e+22, shift = True, wrap = False):

        """
        Build the filament profile using the inputted or recently created filament spine

        Parameters
        ----------
        self: An instance of the radfil_class

        cutdist: float (default=3.0)
            A float indicating how far out from the spine you would like to sample the filament

        pts_mask: numpy.ndarray
            A 2D array masking out any regions from image array you don't want to sample; must be of boolean
            type and the same shape as the image array

        samp_int: integer (default=3)
            An integer indicating how frequently you'd like to make sample cuts
            across the filament.

        save_mask: str,optional
            Saves a figure displaying image mask, the filament spine, the "local width" lines
            and the pixels with the max column density along each local width line

        nobins: boolean (default=False)
            A boolean indicating whether you'd like to bin the profiles.

            A "True" value indicates that no binning will be performed.

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

        Attributes
        ----------
        xfit: numpy.ndarray
            A numpy array containing the x pixel coordinates at which cuts were taken

        yfit: numpy.ndarray
            A numpy array containing the y pixel coordinates at which cuts were taken

        maxcolx: numpy.ndarray
            A numpy array the same size as xfit containing the x coordinate
            of the pixel with the maximum column density along each cut

        maxcoly: numpy.ndarray
            A numpy array the same size as yfit containing the y coordinate
            of the pixel with the maximum column density along each cut

        distpc: numpy.ndarray
            A numpy array containing the physical size of each "local width" cut. Can take
            the mean/median to determine representative width of the filament mask

        xtot: numpy.ndarray
            A 2D numpy array containing the stacked radial distances of each perpendicular cut (in pc)

        ytot: numpy.ndarray
            A 2D numpy array containing the stacked column densities of each perpendicular cut

        masterx: numpy.ndarray
            If nobins=True, will return a concatenated 1D array of xtot (in pc)
            If nobins=False, will interpolate/bin xtot and take median radial distance in each bin

        mastery: numpy.ndarray
            if nobins=True, will return a concatenated 1D array of ytot
            If nobins=False, will interpolate/bin ytot and take median column density in each bin
        """

        # Read cutdist in pc
        if isinstance(cutdist, numbers.Number):
            self.cutdist = float(cutdist) * u.pc

        # Record the setup
        self.nobins = nobins

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
        x, y =profile_tools.curveorder(pixcrd[1], pixcrd[0])
        ## Output for testing.  Remove later.  ###########
        self.xtest, self.ytest = x, y



        # Spline calculation:
        ##set the spline parameters
        k = 5 # spline order ## why 5 when scipy suggested 3?
        nest = -1 # estimate of number of knots needed (-1 = maximal)
        ## find the knot points
        tckp, up, = splprep([x,y], k = k, nest = -1)
        ## evaluate spline
        xfit, yfit = splev(up, tckp)
        xprime, yprime = splev(up, tckp, der=1)
        ## Notice that the result containt points on the spline that are not
        ## evenly sampled.  This might introduce biase when using a single
        ## number `samp_int`.


        # Plot
        fig=plt.figure(figsize=(5,5))
        ax=plt.gca()
        ax.imshow(self.mask, origin='lower', cmap='binary_r', interpolation='none', extent=[0,self.mask.shape[1],0,self.mask.shape[0]])
        ax.plot(xfit, yfit, 'r', label='fit', lw=2, alpha=0.25)
        ax.set_xlim(0, self.mask.shape[1])
        ax.set_ylim(0, self.mask.shape[0])
        self.fig, self.ax = fig, ax

        # Only points within pts_mask AND the original mask are used.
        if (self.pts_mask is not None):
            pts_mask = ((self.pts_mask[np.round(yfit[1:-1:samp_int]).astype(int),
                                       np.round(xfit[1:-1:samp_int]).astype(int)]) &\
                        (self.mask[np.round(yfit[1:-1:samp_int]).astype(int),
                                   np.round(xfit[1:-1:samp_int]).astype(int)]))
        else:
            pts_mask = (self.mask[np.round(yfit[1:-1:samp_int]).astype(int),
                                  np.round(xfit[1:-1:samp_int]).astype(int)])

        # Prepare for extracting the profiles
        self.xfit = xfit[1:-1:samp_int][pts_mask]
        self.yfit = yfit[1:-1:samp_int][pts_mask]
        self.points = np.asarray(zip(self.xfit, self.yfit))
        self.fprime = np.asarray(zip(xprime[1:-1:samp_int][pts_mask], yprime[1:-1:samp_int][pts_mask]))


        # Extract the profiles
        directory_cuts = defaultdict(list)
        for n in range(len(self.points)):
            profile = profile_tools.profile_builder(self, self.points[n], self.fprime[n], shift = shift, wrap = wrap)
            directory_cuts['distance'].append(profile[0]*self.imgscale.to(u.pc).value)
            directory_cuts['profile'].append(profile[1])

        self.directory_cuts = directory_cuts

        # Stack the result and include only points inside `cutdist`.
        xtot, ytot = np.concatenate(directory_cuts['distance']), np.concatenate(directory_cuts['profile'])
        xtot, ytot = xtot[(xtot >= (-self.cutdist/self.imgscale).decompose().value)&\
                          (xtot < (self.cutdist/self.imgscale).decompose().value)],\
                     ytot[(xtot >= (-self.cutdist/self.imgscale).decompose().value)&\
                          (xtot < (self.cutdist/self.imgscale).decompose().value)]
        ## Store the values.
        self.xtot = xtot
        self.ytot = ytot

        if save_mask!=None and isinstance(save_mask,str)==True:
            plt.savefig(save_mask)

        # Bin the profiles (if nobins=False) or stack the profiles (if nobins=True)
        ## This step assumes linear binning.
        if (not nobins):
            ## If the input is the number of bins:
            if isinstance(bins, numbers.Number) and (bins%1 == 0):
                bins = int(round(bins))
                minR, maxR = np.min(self.xtot), np.max(self.ytot)
                bins = np.linspace(minR, maxR, bins+1)
                masterx = bins[:-1]+.5*np.diff(bins)
                mastery = np.asarray([np.median(self.ytot[((self.xtot >= (X-.5*np.diff(bins)[0]))&\
                                               (self.xtot < (X+.5*np.diff(bins)[0])))]) for X in masterx])
            ## If the input is the edges of bins:
            elif isinstance(bins, np.ndarray) and (bins.ndim == 1):
                bins = bins
                masterx = bins[:-1]+.5*np.diff(bins) ## assumes linear binning.
                mastery = np.asarray([np.median(self.ytot[((self.xtot >= (X-.5*np.diff(bins)[0]))&\
                                               (self.xtot < (X+.5*np.diff(bins)[0])))]) for X in masterx])
            ## If the input is wrong.
            else:
                raise TypeError("Bins must be an integer or an 1D numpy.array.")

        ## No binning.
        else:
            masterx = self.xtot
            mastery = self.ytot
            ##### Not sure what std does, yet.
            '''
            std=np.empty((masterx.shape))*np.nan
            nonan=np.where(np.isfinite(mastery)==True)
            masterx=masterx[nonan]
            mastery=mastery[nonan]
            std=std[nonan]
            '''


        # Normalize the profile for better fitting.
        mastery = mastery/norm_constant
        mastery = mastery.astype(float)

        #std=std/norm_constant
        #std=std.astype(float)

        self.masterx=masterx
        self.mastery=mastery
        #self.std=std

        #return image, mask, and spine to original image dimensions without padding
        if self.padsize!=None and self.padsize!=0:
            self.image=self.image[self.padsize:self.image.shape[0]-self.padsize,self.padsize:self.image.shape[1]-self.padsize]
            self.mask=self.mask[self.padsize:self.mask.shape[0]-self.padsize,self.padsize:self.mask.shape[1]-self.padsize]
            self.filspine=self.filspine[self.padsize:self.filspine.shape[0]-self.padsize,self.padsize:self.filspine.shape[1]-self.padsize]

        return self

    def fit_profile(self,model="Gaussian",fitdist=None,subtract_bg=False, bgtype="sloping",bgbounds=None,f_scale=0.5,params=None,filname="Filament Profile Fitting",save_path=None,plot_bg_fit=True,verbose=False):

        """
        Fit a model to the filament's master profile

        Parameters
        ------
        self: An instance of the radfil_class

        model: str or function
            If you'd like to fit the built-in models, choose from either "Plummer" or "Gaussian"
            If you'd like to fit your own model, input your own function

        fitdist: int or float (default=cutdist)
            The radial distance out to which you'd like to fit your profile (must be <= cutdist);

        params: an lmfit.Parameters object, optional
            If you're using your own model, will have to input an lmfit.Parameters instance

        subtract_bg: boolean, optional (default=True)
            Would you like to subtract a background before fitting the profile?

        bgbounds: list, optional
            The lower and upper bounds on your background fitting radius. For instance, if you want to fit a background
            between a radial distance of 3pc and 4 pc, bgbounds=[3,4]

        f_scale: int or float, optional (default=0.5)
            A parameter used in the robust least-squares fitting of the background, with a 'soft_l1' loss function.
            It's defined as the "value of soft margin between inlier and outlier residuals"
            See https://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.least_squares.html for more information

        bg_type: str, optional
            If subtract_bg is true, would you like to subtract a flat background (m=0) or a sloping background (m!=0)?
            Please enter either "flat" or "sloping" as a string

        filname: str,optional
            If you would like to title your plot with the name of the filament

        save_path: str, optional (default: None)
            A string indicating the path and the filename you'd like to save the fits to; if no path is inputted,
            the program will not save the file

        plot_bg_fit: boolean,optional (default=True)
            Would you like to display a plot showing the fit to the background?

        verbose: boolean,optional (default=False)
            Would you like to display the plots?

        Attributes
        ------

        fit_result: lmfit.model.ModelResult
            The result of the least squares fitting of the model
            see https://lmfit.github.io/lmfit-py/model.html#modelresult-attributes for options

        bgline: scipy.optimize.OptimizeResult
            See https://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
            for more info on attributes. Can access the m and b values of the line by typing bgline['x'] which will return an
            array with the first value=m and the second value=b


        """

        if fitdist==None:
            fitdist=self.cutdist

        if isinstance(model, types.FunctionType)==False and model!="Gaussian" and model!="Plummer":
            raise ValueError("Please enter a valid model")

        include=np.where(np.abs(self.masterx)<fitdist)


        #User-inputted model
        if isinstance(model, types.FunctionType)==True:
            if type(params)!=lmfit.parameter.Parameters:
                raise ValueError("Please enter a valid parameter object")
            mod=Model(model)
            params=params

        #Plummer model
        if model=="Plummer":
            mod=Model(plummer)
            params = Parameters()
            params.add('N_0', value=np.max(self.mastery[include]),min=0.0)
            params.add('R_flat', value=np.median(self.distpc)/2.0,min=0.0)
            params.add('p', value=2.0,min=0.0)

        #Gaussian model
        if model=="Gaussian":
            mod=Model(gaussian)
            params=Parameters()
            params.add('amp',value=np.max(self.mastery[include]),min=0)
            params.add('wid',value=np.median(self.distpc)/3.0,min=0)

        #Do background subtraction
        if subtract_bg==True:

            def bgfunc(bgparams, x, y):
                return bgparams[0]*x+bgparams[1]-y

            bgmask=(np.abs(self.masterx)>bgbounds[0]) & (np.abs(self.masterx)<bgbounds[1])
            bgline = least_squares(bgfunc, np.array([0,np.median(self.mastery[bgmask])]),loss='soft_l1',f_scale=f_scale,args=(self.masterx[bgmask],self.mastery[bgmask]))
            self.bgline=bgline
            bgsubtract=self.masterx*bgline['x'][0]+bgline['x'][1]

        else:
            bgsubtract=np.zeros((self.masterx.shape))

        if plot_bg_fit==True:
            fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(10,12))
            plt.suptitle("{}".format(filname),fontsize=20)
            ax[0].scatter(self.masterx, self.mastery, c='b',label="Original Profile",edgecolor='None',s=10,alpha=0.75)
            ax[0].plot(np.linspace(-self.cutdist,+self.cutdist,100),np.linspace(-self.cutdist,+self.cutdist,100)*bgline['x'][0]+bgline['x'][1],'y-',label="Background Fit",lw=4)
            ax[0].set_xlim(-self.cutdist,self.cutdist)
            ax[0].set_xlabel("Radial distance (pc)",fontsize=15)
            ax[0].set_ylabel(r"$\rm H_2 \; Column \; Density \;(10^{22} \; cm^{-2})$",fontsize=15)
            ax[0].legend(loc='best')

            ax[0].text(0.03, 0.95,"{}={:.3f}".format('m',bgline['x'][0]),
                    horizontalalignment='left',verticalalignment='top', fontsize=12, fontweight='bold',transform=ax[0].transAxes)
            ax[0].text(0.03, 0.85,"{}={:.3f}".format('b',bgline['x'][1]),
                    horizontalalignment='left',verticalalignment='top', fontsize=12, fontweight='bold',transform=ax[0].transAxes)

            axnum=1
        else:
            fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(10,10))
            plt.suptitle("{}".format(filname),fontsize=20)
            axnum=0

        result = mod.fit(self.mastery[include]-bgsubtract[include],r=self.masterx[include],params=params)

        ax[axnum].scatter(self.masterx, self.mastery-bgsubtract, c='b',label="Master Profile",edgecolor='None',s=10,alpha=0.75)
        ax[axnum].plot(np.linspace(-fitdist,+fitdist,100),result.model.func(np.linspace(-fitdist,+fitdist,100),*result.params.valuesdict().values()), 'y-',label="Model Fit",lw=4)
        ax[axnum].fill_between(self.masterx, self.mastery-bgsubtract-self.std, self.mastery-bgsubtract+self.std,color='gray',alpha=0.25)
        ax[axnum].set_xlim(-self.cutdist,self.cutdist)
        ax[axnum].axvline(-fitdist,c='k',ls='dashed',alpha=0.3)
        ax[axnum].axvline(+fitdist,c='k',ls='dashed',alpha=0.3)
        ax[axnum].set_xlabel("Radial distance (pc)",fontsize=15)
        ax[axnum].set_ylabel(r"$\rm H_2 \; Column \; Density \;(10^{22} \; cm^{-2})$",fontsize=15)
        ax[axnum].legend(loc='best')

        numparams=len(result.best_values.items())
        textpos=np.linspace(0.95-0.05*numparams,0.95,numparams)
        for i in range(0,numparams):
            ax[axnum].text(0.03, textpos[i],"{}={:.2f}".format(result.best_values.items()[i][0],result.best_values.items()[i][1]),
                    horizontalalignment='left',verticalalignment='top', fontsize=12, fontweight='bold',transform=ax[axnum].transAxes)

        if self.nobins==True:
            ax[axnum+1].scatter(self.masterx[include],result.residual,c='r',label="Residuals",lw=2,edgecolor="None")
        else:
            ax[axnum+1].plot(self.masterx[include],result.residual,'r-',label="Residuals",lw=4)

        ax[axnum+1].set_xlim(-self.cutdist,self.cutdist)
        ax[axnum+1].set_ylim(np.min(result.residual)-0.25,np.max(result.residual)+0.25)
        ax[axnum+1].set_xlabel("Radial distance (pc)",fontsize=15)
        ax[axnum+1].axvline(-fitdist,c='k',ls='dashed',alpha=0.3)
        ax[axnum+1].axvline(+fitdist,c='k',ls='dashed',alpha=0.3)
        ax[axnum+1].legend(loc='best')

        if save_path!=None:
            plt.savefig(save_path)

        if verbose==True:
            plt.show()

        self.fit_result=result

        plt.close('all')

        return self
