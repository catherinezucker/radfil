from .profile_tools import *

class radfil(object):

    """
    Container object which stores the required metadata for building the radial profiles
    
    Parameters
    ------
    image : numpy.ndarray
        A 2D array of the data to be analyzed. 
        
    mask: numpy.ndarray
        A 2D array defining the boundaries of the filament; must be of boolean
        type and the same shape as the image array 
        
    distance : float
        Distance to the filament; must be entered in pc
        
    imgscale : float, optional
        The physical size of each pixel in your image (in pc); only required if
        header is not provided
        
    spine_smooth_factor: integer, optional (default=10)
        The amount of smoothing to be applied to the skeleton spine. This is
        inputted into scipy's splprep function to find the smoothed,
        B-spline representation of the filament spine
        
    cut_separation: float, optional (default=0.5 pc)
        The physical distance between cuts along the filament spine, in pc
        
    filspine: numpy.ndarray, optional
        A 2D array defining the longest path through the filament mask; must
        be of boolean type and the same shape as the img array. 
    """
        
def __init__(self, img, mask, distance, imgscale=None, spine_smooth_factor=None,
                 cut_separation=None, filspine=None):    
                 
    if isinstance(image,np.ndarray)==False or isinstance(mask,np.ndarray)==False :
        raise TypeError("Image and/or mask array is the wrong type; need type np.ndarray")

    if len(image.shape)!= 2 or len(mask.shape)!=2:
        raise TypeError("Image and/or mask array must be 2D.")
        
    if type(distance)!=np.float:
        raise TypeError("Please enter a distance value in pc as a float")
        
    self.image=image
    self.header=header
    self.distance=distance
    
    
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
        fils.create_mask(verbose=False,use_existing_mask=True)

        #do the skeletonization
        fils.medskel(verbose=verbose)

        #find shortest path through skeleton
        analysis=fils.analyze_skeletons(verbose=verbose)
        
        self.filspine=fils.skeleton_longpath
        self.length=np.sum(analysis.lengths)
        self.imgscale=fils.imgscale
        
        return self
        
    def build_profile(self,header,beamwidth=None,verbose=False):
    
    """
    Build the filament profile using the inputted or recently created filament spine 
    
    Parameters
    ------
    header: FITS header
        The header corresponding to the image array. 
        
    beamwidth: float
        A float in units of arcseconds indicating the beamwidth of the image array     
        
    verbose: boolean
        A boolean indicating whether you want to enable FilFinder plotting of filament spine
        
    """
    
    #extract x and y coordinates of filament spine
    pixcrd=np.where(self.filspine)==1)
    x=pixcrd[1]
    y=pixcrd[0]
    pts=np.vstack((x,y)).T
    
    #sort these points by distance along the spine
    xx,yy=filfind_lengths.profile_tools.curveorder(x,y)
    
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

    # evaluate spline, including interpolated points
    xfit,yfit,zfit = splev(u,tckp)
    xprime,yprime,zprime = splev(u,tckp,der=1)
    
    self.xfit=xfit
    self.yfit=yfit
    self.xprime=xprime
    self.yprime=yprime
    

        
        
        

    
    

    
    
