import numpy as np
from fil_finder import fil_finder_2D
from scipy.interpolate import splprep
from scipy.interpolate import splev
import matplotlib.pyplot as plt
import sys
import profile_tools

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
        fig=plt.figure(figsize=(10,10))
        ax=plt.gca()
        plt.imshow(self.mask,origin='lower',zorder=1,cmap='binary_r',interpolation='nearest',extent=[0,self.mask.shape[1],0,self.mask.shape[0]])
        plt.ylim(0,self.image.shape[0])
        plt.xlim(0,self.image.shape[1])
        plt.plot(xx,yy,'b',label='data',lw=2,alpha=0.5)
        plt.plot(xfit,yfit,'r',label='fit',lw=2,alpha=0.5)

        self.xfit=xfit[1:-1:5]
        self.yfit=yfit[1:-1:5]
        self.points=xfit[1:-1:5]
        self.fprime=yprime[1:-1:5]/xprime[1:-1:5]
        self.m=-1.0/(yprime[1:-1:5]/xprime[1:-1:5])
              
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
        
        self.masterx=masterx
        self.mastery=mastery
        
        return self
        

    
        

        

        
    

        
        
        

    
    

    
    
