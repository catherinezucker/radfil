import numpy as np
import types
from fil_finder import fil_finder_2D
from scipy.interpolate import splprep
from scipy.interpolate import splev
import matplotlib.pyplot as plt
import lmfit
from lmfit.models import Model
from lmfit import Parameters
import sys
import profile_tools


def plummer(r,N_0,R_flat,p):
    return (N_0)/(1+(r/R_flat)**2)**((p-1)/2.0)
            
def plummer_bg(r,N_0,R_flat,p,bg):
    return (N_0)/(1+(r/R_flat)**2)**((p-1)/2.0) + bg
            
def gaussian(x, amp, wid):
    return (amp) * np.exp(-1 * np.power(x, 2) / (2 * np.power(wid, 2)))
            
def gaussian_bg(x, amp, wid, bg):
    return (amp) * np.exp(-1 * np.power(x, 2) / (2 * np.power(wid, 2))) + bg


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
        
    distance : float
        Distance to the filament; must be entered in pc
        
    header : astropy.io.fits.Header
        The header corresponding to the image array
        
    spine_smooth_factor: integer, optional (default=10)
        The amount of smoothing to be applied to the skeleton spine. This is
        inputted into scipy's splprep function to find the smoothed,
        B-spline representation of the filament spine
        
    cut_separation: float, optional (default=0.5 pc)
        The physical distance between cuts along the filament spine, in pc
        
    filspine: numpy.ndarray, optional
        A 2D array defining the longest path through the filament mask; must
        be of boolean type and the same shape as the img array. Can also create
        your own with the FilFinder package using the "make_fil_spine" method. 
    """
        
    def __init__(self, image, mask, header, distance, spine_smooth_factor=None,
                    cut_separation=None, filspine=None):    
                 
        if isinstance(image,np.ndarray)==False or isinstance(mask,np.ndarray)==False :
            raise TypeError("Image and/or mask array is the wrong type; need type np.ndarray")

        if len(image.shape)!= 2 or len(mask.shape)!=2:
            raise TypeError("Image and/or mask array must be 2D.")
        
        if type(distance)!=np.float:
            raise TypeError("Please enter a distance value in pc as a float")
        
        self.image=image
        self.mask=mask
        self.distance=distance
        self.spine_smooth_factor=spine_smooth_factor
        self.cut_separation=cut_separation
        self.filspine=filspine
        self.imgscale = header["CDELT2"] * (np.pi / 180.0) * distance


    
    def make_fil_spine(self,header,beamwidth=None,verbose=False):

        """
        Create filament spine using the FilFinder package 'shortest path' option
    
        Parameters
        ------
        header: FITS header
            The header corresponding to the image array. 
        
        beamwidth: float
            A float in units of arcseconds indicating the beamwidth of the image array     
        
        verbose: boolean
            A boolean indicating whether you want to enable FilFinder plotting of filament spine
        """
    
        #Create a filament spine with the FilFinder package, using your image array and mask array 
        
        if beamwidth is None:
            raise ValueError("Beamwidth must be provided to create filament spine")
            
        if type(beamwidth)!=np.float:
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
        
    def build_profile(self, cutdist=3.0,plot_cuts=True,plot_samples=False):
    
        """
        Build the filament profile using the inputted or recently created filament spine 
    
        Parameters
        ------
        self: An instance of the radfil_class
        
        plot_cuts: boolean (default=True)
            A boolean indicating whether you want to plot the local width lines across the spine
        
        plot_samples: boolean (default=False)
            A boolean indicating whether you want to plot the points at which the profiles are sampled 
        
        show_plots: boolean (default=True)
            A boolean indicating whether you want to display the plots    
        
        """
    
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
        s=10 # smoothness parameter
        k=5 # spline order
        nest=-1 # estimate of number of knots needed (-1 = maximal)

        # find the knot points
        tckp,u = splprep([x,y,z],k=k,nest=-1)

        #evaluate spline, including interpolated points
        xfit,yfit,zfit = splev(u,tckp)
        xprime,yprime,zprime = splev(u,tckp,der=1)
        
        #build plot
        fig=plt.figure(figsize=(8,8))
        ax=plt.gca()
        plt.imshow(self.mask,origin='lower',zorder=1,cmap='binary_r',interpolation='nearest',extent=[0,self.mask.shape[1],0,self.mask.shape[0]])
        plt.ylim(0,self.image.shape[0])
        plt.xlim(0,self.image.shape[1])
        #plt.plot(xx,yy,'b',label='data',lw=2,alpha=0.5)
        plt.plot(xfit,yfit,'r',label='fit',lw=2,alpha=0.5)

        self.xfit=xfit[1:-1:3]
        self.yfit=yfit[1:-1:3]
        self.points=xfit[1:-1:3]
        self.fprime=yprime[1:-1:3]/xprime[1:-1:3]
        self.m=-1.0/(yprime[1:-1:3]/xprime[1:-1:3])
              
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
        
        delta=np.clip((cutdist+1)/self.imgscale,(np.max(distpc)+1)/self.imgscale,np.inf)
        deltamax=[]
        for i in range(0,self.points.shape[0]):
            arr1=np.array([[1,1],[1,-self.m[i]**2]])
            arr2=np.array([delta**2,0])
            solved = np.linalg.solve(arr1, arr2)
            y=np.sqrt(solved[0])+self.maxcoly[i]
            x=np.sqrt(solved[1])+self.maxcolx[i]
            deltamax.append(np.sqrt(np.abs(solved[1])))
            
        self.deltamax=deltamax
        
        xtot,ytot=profile_tools.get_radial_prof(self,maxcolx,maxcoly,ax=ax,cutdist=3.0,plot_max=True,plot_samples=False)
        
        masterx,mastery,std=profile_tools.make_master_prof(xtot,ytot)
        mastery=mastery/10**22
        mastery=mastery.astype(float)
        
        std=std/10**22
        std=std.astype(float)
        
        self.masterx=masterx
        self.mastery=mastery
        self.std=std
        
        self.cutdist=cutdist
        
        return self
        
    def fit_profile(self,model="Gaussian",fitdist=None,params=None,fit_bg=False,filname="Filament Profile Fitting"):
    
        """
        Fit a model to the filament's master profile 
    
        Parameters
        ------
        self: An instance of the radfil_class
        
        model: str or function
            If you'd like to fit the built-in models, choose from either "Plummer" or "Gaussian"
            If you'd like to fit your own model, input your function
            
        fitdist: the radial distance you'd like to fit to (must be <= cutdist); default=cutdist
            
        params: an lmfit.Parameters object, optional
            If you're using your own model, will have to input an lmfit.Parameters instance
        
        fit_bg= boolean, optional (default=False)
            Would you like to fit a background parameter as part of the built-in Plummer-like/Gaussian models?
            
        filname= str,optional
            If you would like to title your plot with the name of the filament
        
        """
        
        if fitdist==None:
            fitdist=self.cutdist
                
        if isinstance(model, types.FunctionType)==False and model!="Gaussian" and model!="Plummer":            
            raise ValueError("Please enter a valid model")
          
        if isinstance(model, types.FunctionType)==True:
        
            if type(params)!=lmfit.parameter.Parameters:
                raise ValueError("Please enter a valid parameter object")
                
            mod=Model(model)
            params=params
            
        include=np.where(np.abs(self.masterx)<fitdist)
              
        if model=="Plummer" and fit_bg==False:
            mod=Model(plummer)
            params = Parameters()
            params.add('N_0', value=np.max(self.mastery[include]))
            params.add('R_flat', value=np.median(self.distpc)/2.0)
            params.add('p', value=2.0)
            
        if model=="Plummer" and fit_bg==True:
            mod=Model(plummer_bg)
            params = Parameters()
            params.add('N_0', value=np.max(self.mastery[include])-np.min(self.mastery[include]))
            params.add('R_flat', value=np.median(self.distpc)/2.0)
            params.add('p', value=2.0)
            params.add('bg', value=np.min(self.mastery[include]))
            
        if model=="Gaussian" and fit_bg==False:
            mod=Model(gaussian)
            params=Parameters()
            params.add('amp',value=np.max(self.mastery[include]))
            params.add('width',value=np.median(self.distpc))
            
        if model=="Gaussian" and fit_bg==True:
            mod=Model(gaussian_bg)
            params=Parameters()
            params.add('amp',value=np.max(self.mastery[include])-np.min(self.mastery[include]))
            params.add('width',value=np.median(self.distpc))
            params.add('bg',value=np.min(self.mastery[include]))
            
            
        result = mod.fit(self.mastery[include],r=self.masterx[include],params=params)
    
        fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(20,5))
        plt.suptitle("{}".format(filname),fontsize=30)
                
        ax[0].scatter(self.masterx, self.mastery, c='b',label="Data",edgecolor='None')
        ax[0].plot(self.masterx[include], result.best_fit, 'y-',label="Model Fit",lw=2)
        ax[0].fill_between(self.masterx, self.mastery-self.std, self.mastery+self.std,color='gray',alpha=0.25)
        ax[0].set_xlim(-self.cutdist,self.cutdist)
        ax[0].axvline(-fitdist,c='k',ls='dashed',alpha=0.3)
        ax[0].axvline(+fitdist,c='k',ls='dashed',alpha=0.3)
        ax[0].set_xlabel("Radial distance (pc)",fontsize=20)
        ax[0].set_ylabel(r"$\rm H_2 \; Column \; Density \;(10^{22} \; cm^{-2})$",fontsize=20)
        
        numparams=len(result.best_values.items())
        textpos=np.linspace(0.95-0.05*numparams,0.95,numparams)
        for i in range(0,numparams):
            ax[0].text(0.03, textpos[i],"{}={:.2f}".format(result.best_values.items()[i][0],result.best_values.items()[i][1]), 
                        horizontalalignment='left',verticalalignment='top', fontsize=15, fontweight='bold',transform=ax[0].transAxes)
    
        ax[1].plot(self.masterx[include], result.residual, 'r-',label="Residuals",lw=2)
        ax[1].set_xlim(-fitdist,fitdist)
        ax[1].set_ylim(-0.25,0.25)
        ax[1].set_xlabel("Radial distance (pc)",fontsize=20)
        ax[1].legend()        
        
        return self
       
        

        

        
    

        
        
        

    
    

    
    
