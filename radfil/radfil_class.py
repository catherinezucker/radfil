import numpy as np
import types
from scipy.interpolate import splprep
from scipy.interpolate import splev
import matplotlib.pyplot as plt
import lmfit
from lmfit.models import Model
from lmfit import Parameters
import sys
from radfil import profile_tools
from astropy import units as u 
import math
from matplotlib.colors import LogNorm
from scipy.optimize import least_squares


#taken from Eric Koch's filfinder package; linked here: 
#https://github.com/e-koch/FilFinder/blob/7bca14d597f0d0911628384ccf48eb874a7f8380/fil_finder/utilities.py
def padwithzeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    if pad_width[1] > 0:
        vector[-pad_width[1]:] = 0
    return vector

def plummer(r,N_0,R_flat,p):
    return (N_0)/(1+(r/R_flat)**2)**((p-1)/2.0)
            
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
        
    distance : float or int
        Distance to the filament; must be entered in pc
        
    filspine: numpy.ndarray, optional
        A 2D array defining the longest path through the filament mask; must
        be of boolean type and the same shape as the img array. Can also create
        your own with the FilFinder package using the "make_fil_spine" method. 
        
    padsize: int, optional
        In cases in which the filament is too close to the border of the image,
        might need to pad in order for RadFil to successfully run. 
        
    Attributes
    ----------
        imgscale : float
           The image scale in pc of each pixel
    """
        
    def __init__(self, image, mask, header, distance, filspine=None,padsize=None):    
                 
        if isinstance(image,np.ndarray)==False or isinstance(mask,np.ndarray)==False :
            raise TypeError("Image and/or mask array is the wrong type; need type numpy.ndarray")

        if len(image.shape)!= 2 or len(mask.shape)!=2:
            raise TypeError("Image and/or mask array must be 2D.")
        
        if isinstance(distance,float)== False and isinstance(distance,int)==False:
            raise TypeError("Please enter a distance value in pc as an int or float")
        
        self.image=image
        self.mask=mask
        self.header=header
        self.distance=distance
        self.filspine=filspine
        self.imgscale = header["CDELT2"] * (np.pi / 180.0) * distance
        self.padsize=padsize
        
        if padsize!=None and isinstance(padsize,int)==True:
            self.image=np.pad(self.image,padsize,padwithzeros)
            self.mask=np.pad(self.mask,padsize,padwithzeros)
            
            if self.filspine!=None:
                self.filspine=np.pad(self.filspine,padsize,padwithzeros)

    
    def make_fil_spine(self,beamwidth=None,verbose=False):

        """
        Create filament spine using the FilFinder package 'shortest path' option
    
        Parameters:
         ----------
        beamwidth: float
            A float in units of arcseconds indicating the beamwidth of the image array     
        
        verbose: boolean
            A boolean indicating whether you want to enable FilFinder plotting of filament spine
            
        Attributes
        ----------
        filspine : numpy.ndarray
           A 2D array of 1s and 0s defining the longest path through the filament mask
        """
    
        #Create a filament spine with the FilFinder package, using your image array and mask array 
        from fil_finder import fil_finder_2D
        
        if isinstance(beamwidth,float)==False:
            raise TypeError("Beamwidth must be provided to create filament spine")
            
        fils=fil_finder_2D(self.image,self.header,beamwidth=beamwidth*u.arcsec,distance=self.distance*u.pc,mask=self.mask)
        
        #create the mask
        fils.create_mask(verbose=verbose,use_existing_mask=True)

        #do the skeletonization
        fils.medskel(verbose=verbose)

        #find shortest path through skeleton
        analysis=fils.analyze_skeletons(verbose=verbose)
        
        self.filspine=fils.skeleton_longpath
        self.length=np.sum(analysis.lengths)
        self.imgscale=fils.imgscale
        
        return self
        
    def build_profile(self,cutdist=3.0,pts_mask=None,samp_int=3,numbins=120,save_mask=None,save_cuts=None,nobins=False,norm_constant=1e+22):
    
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
            An integer indicating how frequently you'd like to make sample cuts across the filament
            
        save_mask: str,optional
            Saves a figure displaying image mask, the filament spine, the "local width" lines
            and the pixels with the max column density along each local width line
        
        save_cuts: str,optional
            Saves a figure containing the profiles for all the cuts along the filament,
            along with the master profile calculated by taking the average, median profile. 
            Will only save if nobins=False
            
        nobins: boolean (default=False)
            A boolean indicating whether you'd like to bin the profiles. 
            
        numbins: int, optional (default=120)
            The number of bins you'd like to divide the whole profile (-cutdist to +cutdist) into, assuming nobins=False. 
            If false, all of the individual profiles are binned by distance from r=0 pc and then the median column density
            in each of these bins is taken to determine the master profile
            
        norm_constant: float, optional (default=1e+22)
            Would you like to normalize your column densites (or flux) values by some normalization constant? If so
            enter it as a float or int 
            
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
        
        self.cutdist=cutdist
        self.nobins=nobins

        #Pad the pts_mask if it exists and padsize!=0
        if self.padsize!=None and self.padsize!=0 and pts_mask!=None:
            pts_mask=np.pad(pts_mask,self.padsize,padwithzeros)

        #extract x and y coordinates of filament spine
        pixcrd=np.where(self.filspine==1)
        x=pixcrd[1]
        y=pixcrd[0]
            
        #sort these points by distance along the spine
        xx,yy=profile_tools.curveorder(x,y)
    
        #parameterize the filament spine by x values, y values, and order along the spine "t"
        t=np.arange(0,xx.shape[0],1)
        x = xx
        y = yy
        z = t

        #set the spline parameters
        k=5 # spline order
        nest=-1 # estimate of number of knots needed (-1 = maximal)

        # find the knot points
        tckp,u = splprep([x,y,z],k=k,nest=-1)

        #evaluate spline
        xfit,yfit,zfit = splev(u,tckp)
        xprime,yprime,zprime = splev(u,tckp,der=1)
        
        #build plot
        fig=plt.figure(figsize=(5,5))
        ax=plt.gca()
        plt.imshow(self.mask,origin='lower',zorder=1,cmap='binary_r',interpolation='nearest',extent=[0,self.mask.shape[1],0,self.mask.shape[0]])
        plt.ylim(0,self.image.shape[0])
        plt.xlim(0,self.image.shape[1])
        plt.plot(xfit,yfit,'r',label='fit',lw=2,alpha=0.25)
                                
        #If pts_mask!=None, only sample points which fall inside pts_mask. Otherwise, sample along entire spine
        if pts_mask!=None:
            pts_mask=np.where(pts_mask[yfit[1:-1:samp_int].astype(int),xfit[1:-1:samp_int].astype(int)]==1)
        else:
            pts_mask=np.ones((yfit[1:-1:samp_int].shape)).astype(bool)

        self.xfit=xfit[1:-1:samp_int][pts_mask]
        self.yfit=yfit[1:-1:samp_int][pts_mask]
        self.points=xfit[1:-1:samp_int][pts_mask]
        self.fprime=yprime[1:-1:samp_int][pts_mask]/xprime[1:-1:samp_int][pts_mask]
        self.m=-1.0/(yprime[1:-1:samp_int][pts_mask]/xprime[1:-1:samp_int][pts_mask])
              
        delta=1.0
        deltax=[]
        for i in range(0,self.points.shape[0]):
            arr1=np.array([[1,1],[1,-self.m[i]**2]])
            arr2=np.array([delta**2,0])
            solved = np.linalg.solve(arr1, arr2)
            y=np.sqrt(solved[0])+self.yfit[i]
            x=np.sqrt(solved[1])+self.xfit[i]
            deltax.append(np.sqrt(np.abs(solved[1])))
            
        self.deltax=np.array(deltax)
        
        leftx,rightx,lefty,righty,distpc=profile_tools.maskbounds(self,ax=ax)
        
        self.distpc=distpc
                
        if leftx.shape[0]!= self.points.shape[0]:
            raise AssertionError("Missing point")
        
        maxcolx,maxcoly=profile_tools.max_intensity(self,leftx,rightx,lefty,righty,ax=ax)
        
        self.maxcolx=maxcolx
        self.maxcoly=maxcoly
        
        deltamax=[]
        for i in range(0,self.points.shape[0]):
            delta=np.clip((cutdist)/self.imgscale,distpc[i]/self.imgscale,np.inf)
            arr1=np.array([[1,1],[1,-self.m[i]**2]])
            arr2=np.array([delta**2,0])
            solved = np.linalg.solve(arr1, arr2)
            y=np.sqrt(solved[0])+self.maxcoly[i]
            x=np.sqrt(solved[1])+self.maxcolx[i]
            deltamax.append(np.sqrt(np.abs(solved[1])))
            
        self.deltamax=deltamax
        
        xtot,ytot,samplesx,samplesy=profile_tools.get_radial_prof(self,maxcolx,maxcoly,ax=ax,cutdist=cutdist)
        
        self.xtot=xtot
        self.ytot=ytot
        
        if save_mask!=None and isinstance(save_mask,str)==True:
            plt.savefig(save_mask)
        
        #Bin the profiles (if nobins=False) or stack the profiles (if nobins=True)
        if nobins==False:
            masterx,mastery,std=profile_tools.make_master_prof(xtot,ytot,cutdist=cutdist,numbins=numbins)
        else:
            masterx=np.hstack((xtot))
            mastery=np.hstack((ytot))
            std=np.empty((masterx.shape))*np.nan
            nonan=np.where(np.isfinite(mastery)==True)
            masterx=masterx[nonan]
            mastery=mastery[nonan]
            std=std[nonan]

        if save_cuts!=None and isinstance(save_cuts,str)==True:
            plt.savefig(save_cuts)
        
        mastery=mastery/norm_constant
        mastery=mastery.astype(float)        
        
        std=std/norm_constant
        std=std.astype(float)
        
        self.masterx=masterx
        self.mastery=mastery
        self.std=std
            
        #return image, mask, and spine to original image dimensions without padding
        if self.padsize!=None and self.padsize!=0:
            self.image=self.image[self.padsize:self.image.shape[0]-self.padsize,self.padsize:self.image.shape[1]-self.padsize]
            self.mask=self.mask[self.padsize:self.mask.shape[0]-self.padsize,self.padsize:self.mask.shape[1]-self.padsize]
            self.filspine=self.filspine[self.padsize:self.filspine.shape[0]-self.padsize,self.padsize:self.filspine.shape[1]-self.padsize]

        return self
        
    def fit_profile(self,model="Gaussian",fitdist=None,subtract_bg=False, bgtype="sloping",bgbounds=None,f_scale=0.5,params=None,filname="Filament Profile Fitting",save_path=None,plot_bg_fit=True):
    
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
            
        filname= str,optional
            If you would like to title your plot with the name of the filament
            
        save_path=str, optional (default: None)
            A string indicating the path and the filename you'd like to save the fits to; if no path is inputted,
            the program will not save the file 
        
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
                    
        self.fit_result=result
        
        plt.close('all')
        
        return self
       
        

        

        
    

        
        
        

    
    

    
    
