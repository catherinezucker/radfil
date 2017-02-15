import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import scipy
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d


def curveorder(x,y):
    """
    Sort pixels that make up the filament spine by the order in which they appear along the curve
    Code taken from http://stackoverflow.com/questions/37742358/sorting-points-to-form-a-continuous-line
    
    Parameters:
    
    x: numpy.ndarray
       A 1d array specifying the x coordinates of the pixels defining your filament spine
    y: numpy.ndarray
       A 1d array specifying the y coordinates of the pixels defining your filament spine
       
    Returns:
    xx, yy: the sorted x and y arrays given above
    
    """ 
    
    pts=np.vstack((x,y)).T
    
    clf = NearestNeighbors(2).fit(pts)
    G = clf.kneighbors_graph()


    T = nx.from_scipy_sparse_matrix(G)

    order = list(nx.dfs_preorder_nodes(T, 0))

    xx = x[order]
    yy = y[order]

    paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(pts))]

    mindist = np.inf
    minidx = 0

    for i in range(len(pts)):
        p = paths[i]           # order of nodes
        ordered = pts[p]    # ordered nodes
        # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
        cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
        if cost < mindist:
            mindist = cost
            minidx = i
        
    opt_order = paths[minidx]

    xx = x[opt_order]
    yy = y[opt_order]
    
    return(xx,yy)
    
    
def maskbounds(radobj,ax):

    """
    Determine where the perpendicular cuts touch the mask on either side of the spine. These define
    each of the "local width lines" perpendicular to the smoothed filament spine 
    
    Parameters:
    -----------
       
    radobj:
       An instance of the radfil_class which now contain the following relevant attributes: 
       
        points: numpy.ndarray
            A 1d array specifying the x coordinates of the pixels at which you want to sample the widths
    
        imgscale: float
            The physical size of each pixel in pc
        
        image : numpy.ndarray
            A 2D array of the data to be analyzed. 
       
        mask: numpy.ndarray
            A 2D array defining the boundaries of the filament; must be of boolean
            type and the same shape as the image array 
        
        xfit: numpy.ndarray
            The x coordinates defining the smoothed B-spline representation of the filament spine
        
        yfit: numpy.ndarray
            The y coordinates defining the smoothed B-spline representation of the filament spine
        
        xprime: numpy.ndarray
            The first derivative of xfit along the smoothed spine
    
        yprime: numpy.ndarray
            The first derivative of yfit along the smoothed spine
        
    ax: matplotlib.axes._subplots.AxesSubplot

    Returns:
    -----------

    leftx: numpy.ndarray
        1D numpy array containing x coordinates at the LHS of each width line
        
    rightx: numpy.ndarray
        1D numpy array containing the x coordinates at the RHS of each width line
        
    lefty: numpy.ndarray
        1D numpy array containing the y coordinates at the LHS of each width line
        
    righty: numpy.ndarray
        1D numpy array containing the y coordinates at the RHS of each width line
        
    distpc: numpy.ndarray
        1D array containing the width of each line in pc
    
    """ 
    
    dist=[]
    distpc=[]
    finalpts=[]
    leftx=[]
    rightx=[]
    lefty=[]
    righty=[]
            
    for i in range(0,len(radobj.points)):
    
        #Instantiate the parameters defining the equation of the line perpendicular to the tangent at the given point
        a=radobj.points[i]
        fa=radobj.yfit[i]
        fprime=radobj.fprime[i]
        m=radobj.m[i]
        deltax=radobj.deltax[i]

        #Iteratively increase the length of the width line outwards from the tangent point until you reach both sides of filament mask
        linerange=np.arange(0,10000,1)
                    
        for line in linerange:
        
            findxbound=np.linspace(a-line*deltax,a+line*deltax,10000)
            findybound=np.array(fa+(-1.0/fprime)*(findxbound-a)) # tangent
            
            if (np.min(findxbound)<0) or (np.max(findxbound)>radobj.image.shape[1]-1) or (np.min(findybound)<0) or (np.max(findybound)>radobj.image.shape[0]-1):
                raise IndexError("Not enough buffer to determine local width of mask. Please rerun with larger padsize using built in padsize argument")
            
            boundarr=radobj.mask[findybound.astype(int),findxbound.astype(int)]
            halved=findxbound.shape[0]/2
            
            #Tests case when mask is entirely confined to either LHS or RHS of the spine
            if ((np.any(boundarr[:halved])==True and np.count_nonzero(boundarr[halved:])==0)==True or (np.any(boundarr[halved:])==True and np.count_nonzero(boundarr[:halved])==0)==True):
                    
                lhs = np.where(boundarr==1)[0][0]
                rhs = np.where(boundarr==1)[0][-1]
                
                fin_x=findxbound[lhs:rhs+1]
                fin_tanperp=findybound[lhs:rhs+1]
                
                leftx.append(fin_x[0])
                rightx.append(fin_x[-1])
    
                lefty.append(fin_tanperp[0])
                righty.append(fin_tanperp[-1])            
                                                    
                dist = math.hypot(fin_x[0] - fin_x[-1], fin_tanperp[0] - fin_tanperp[-1])
                distpc.append(dist*radobj.imgscale)
                
                ax.plot(fin_x,fin_tanperp,ls='solid',c='r',alpha=1.0,zorder=2)  
                                           
                break
                
            #Tests case when mask is on both sides of spine
            if (np.all(boundarr[:halved])==0 and np.all(boundarr[halved:])==0 and np.any(boundarr)==1)==True:

                lhs = np.where(boundarr[:halved]==0)[0][-1]
                rhs = np.where(boundarr[halved:]==0)[0][0]
                
                fin_x=findxbound[lhs:rhs+halved]
                fin_tanperp=findybound[lhs:rhs+halved]
                                    
                leftx.append(fin_x[0])
                rightx.append(fin_x[-1])
    
                lefty.append(fin_tanperp[0])
                righty.append(fin_tanperp[-1])            
                                                    
                dist = math.hypot(fin_x[0] - fin_x[-1], fin_tanperp[0] - fin_tanperp[-1])
                distpc.append(dist*radobj.imgscale)
                    
                ax.plot(fin_x,fin_tanperp,ls='solid',c='r',alpha=1.0,zorder=2)   
                                        
                break 
                
    return (np.array(leftx),np.array(rightx),np.array(lefty),np.array(righty),np.array(distpc))
    
def max_intensity(radobj,leftx,rightx,lefty,righty,ax):

    """
    Determine the pixel along each of the local width lines with the maximum column density value
    
    Parameters:
    radobj:
       An instance of the radfil_class which contains all the relevant attributes described in maskbounds function above
       
    leftx: numpy.ndarray
        1D array containing x coordinates defining the LHS of each width line
        
    rightx: numpy.ndarray
        1D arraying containing the x coordinates defining the RHS of each width line
        
    lefty: numpy.ndarray
        1D array containing the y coordinates defining the LHS of each width line
        
    righty: numpy.ndarray 
        1D array containing the y coordinates defining the RHS of each width line    
        
    ax: matplotlib.axes._subplots.AxesSubplot
        The axes for the current figure
        
    Returns: 
    maxcolx: numpy.ndarray
        1D array containing the x coordinates of the pixels with the maximum column density along each width line
    
    maxcoly:
        1D array containing the y coordinates of the pixels with the maximum column density along each width line
        
    """
    maxcolx=[]
    maxcoly=[]

    for i in range(0,len(radobj.points)):
    
        #Instantiate the parameters defining the equation of the line perpendicular to the tangent at the given point
        a=radobj.points[i]
        fa=radobj.yfit[i]
        fprime=radobj.fprime[i]
        m=radobj.m[i]
                
        #Sample the line along the entire local width
        findxbound=np.linspace(np.min([leftx[i],rightx[i]]),np.max([leftx[i],rightx[i]]),10000)
        findybound=np.array(fa+(-1.0/fprime)*(findxbound-a)) # tangent line
        
        #Determine unique pixel values along the local width
        unique,indices,inverse,counts=np.unique(radobj.image[findybound.astype(int),findxbound.astype(int)],return_index=True,return_inverse=True,return_counts=True)
    
        uniquex=findxbound.astype(int)[indices]
        uniquey=findybound.astype(int)[indices]
    
        order=np.argsort(indices)
        indices=indices[order]
        counts=counts[order]
        
        #if same pixel value is sampled multiple times, extract only one of each instance, closest to the center of the path through each pixel
        extract=[]
        for j in range(0,len(counts)):
            extract.append(np.sum(counts[0:j])+counts[j]/2)
        
        coldensity=radobj.image[findybound.astype(int)[extract],findxbound.astype(int)[extract]]
    
        #of the unique pixels, determine one with maximum column density value
        argmax=np.argmax(coldensity)
        maxcolx.append(int(findxbound[extract[argmax]]))
        maxcoly.append(int(findybound[extract[argmax]]))
        
        #mark the pixel with the maximum column density
        ax.scatter(int(findxbound[extract[argmax]])+0.5,int(findybound[extract[argmax]])+0.5,c='blue',edgecolor="None",zorder=4,s=20,marker='o',alpha=0.5)
 
    return np.array(maxcolx),np.array(maxcoly)
        
def get_radial_prof(radobj,maxcolx,maxcoly,ax=None,cutdist=3.0):

    """
    Return the radial profile along each perpendicular cut across the spine, shifted to the peak column density
    
    Parameters:
    radobj:
       An instance of the radfil_class
       
    maxcolx: numpy.ndarray
        1D array containing x coordinates of the pixels with the maximum column density along each local width line
        
    maxcoly: numpy.ndarray
        1D array containing y coordinates of the pixels with the maximum column density along each local width line
        
    cutdist: float
        The maximum radial distance out to which you want to sample the profile. Must be in pc. 
        
    ax: matplotlib.axes._subplots.AxesSubplot
        The axes for the current figure
        
    Returns: 
    xtot: numpy.ndarray
        2D array containing the radial distances from the peak column density pixel (defined to be r=0) for all perpendicular cuts
    
    ytot: numpy.ndarray
        2D array containing the column densities corresponding to the radial distances in xtot for all perpendicular cuts
        
     """
    
    xtot=[]
    ytot=[]
    samplesx=[]
    samplesy=[]
    
    indexedarr=np.cumsum(np.ones((radobj.image.shape))).reshape((radobj.image.shape))
    
    for i in range(0,len(radobj.points)):
        
        #Instantiate the parameters defining the equation of the line perpendicular to the tangent at the given point
        a=radobj.points[i]
        fa=radobj.yfit[i]
        fprime=radobj.fprime[i]
        m=radobj.m[i]
        deltamax=radobj.deltamax[i]
                
        findxbound=np.linspace(a-deltamax,a+deltamax,1000000)
        findybound=np.array(fa+(-1.0/fprime)*(findxbound-a)) # tangent

        unique,indices,inverse,counts=np.unique(indexedarr[findybound.astype(int),findxbound.astype(int)],return_index=True,return_inverse=True,return_counts=True)

        uniquex=findxbound.astype(int)[indices]
        uniquey=findybound.astype(int)[indices]

        order=np.argsort(indices)
        indices=indices[order]
        counts=counts[order]

        extract=[]
        for j in range(0,len(counts)):
            extract.append(np.sum(counts[0:j])+counts[j]/2)

        coldensity=radobj.image[findybound.astype(int)[extract],findxbound.astype(int)[extract]]

        #build radial distance array out from the pixel of maximum column density 
        radialdist=[]
        center = np.where(((findxbound.astype(int)[extract]==maxcolx[i]) & (findybound.astype(int)[extract]==maxcoly[i]))==True)[0]
        centerx=findxbound[extract][center]
        centery=findybound[extract][center]
        for pt in zip(findxbound[extract],findybound[extract]):
            radialdist.append(math.hypot(pt[0] - centerx, pt[1] - centery))

        #convert to physical units
        radialdist=np.array(radialdist)*radobj.imgscale
 
        #designate left side of spine as negative side
        leftspine=np.where(findxbound[extract]<centerx)
        radialdist[leftspine[0]]=radialdist[leftspine[0]]*-1

        #delete points beyond the cut distance
        deletepts=np.where(np.abs(radialdist)>cutdist)
        radialdist=np.delete(radialdist,deletepts)
        coldensity=np.delete(coldensity,deletepts)
        extract=np.delete(extract,deletepts)

        xtot.append(radialdist)
        ytot.append(coldensity)

        #Sanity Check
        if indexedarr[findybound.astype(int)[extract],findxbound.astype(int)[extract]].shape[0]!=np.unique(indexedarr[findybound.astype(int)[extract],findxbound.astype(int)[extract]]).shape[0]:
            raise AssertionError("Profile Has Repeat Column Density Value")

        samplesx.append(findxbound[extract])
        samplesy.append(findybound[extract])
        
    return np.array(xtot),np.array(ytot),np.array(samplesx),np.array(samplesy)
    
    
def make_master_prof(xtot,ytot,cutdist=3.0,numbins=120):

    """
    Linearly interpolate profiles on to finer grid, bin the interpolated profiles, and take the median in each bin to determine "master" profile
    
    Parameters:
    ------------
    xtot: numpy.ndarray
        2D array containing the radial distances from the peak column density pixel (defined to be r=0) for all perpendicular cuts
       
    ytot: numpy.ndarray
        2D array containing the column densities corresponding to the radial distances stored in xaxis for all perpendicular cuts
        
    cutdist: float
        A float indicating how far out you'd like to sample the profile
        
    numbins: int, optional (default=120)
        The number of bins you'd like to divide the whole profile (-cutdist to +cutdist) into. All of the individual profiles are binned
        by distance from r=0 pc and then the median in each of these bins is taken to determine the master profile
        
    Returns: 
    ---------
    masterx: numpy.ndarray
        1D array containing the median radial distance in each bin
    
    mastery: numpy.ndarray
        1D array containing the median column density each bin
        
    std: numpy.ndarray
        1D array containing the standard deviation of the column density in each bin
    """
    
    fig=plt.figure(figsize=(5,5))
    
    #Interpolate linearly between sample points
    xinterp=[]
    yinterp=[]
    
    for i in range(0,len(xtot)):
        sample=np.linspace(float("%.2f" % (int(np.min(xtot[i])*100)/float(100))),float("%.2f" % (int(np.max(xtot[i])*100)/float(100))),1000) #truncate to 2 decimal places
        f = interp1d(xtot[i], ytot[i])
        xresamp=sample
        yresamp=f(sample)
        xinterp.append(xresamp)
        yinterp.append(yresamp)
        plt.plot(xtot[i],ytot[i],c='k',alpha=0.1)
        
    plt.xlim(-cutdist,cutdist)
    plt.xlabel("Radial Distance (pc)")
    plt.ylabel(r"$\rm H_2 \; Column \; Density \;(cm^{-2})$")

    xinterp=np.hstack(xinterp)
    yinterp=np.hstack(yinterp)
    
    #bin the interpolated samples
    bins=np.linspace(-cutdist,cutdist,numbins)
    binorder=np.argsort(xinterp)
    xinterp=xinterp[binorder]
    yinterp=yinterp[binorder]
    inds = np.digitize(xinterp, bins)

    masterx=[]
    mastery=[]
    std=[]
    for i in range(0,np.max(inds)):
        masterx.append(np.nanmedian(xinterp[np.where(inds==i)]))
        mastery.append(np.nanmedian(yinterp[np.where(inds==i)]))
        std.append(np.nanstd(yinterp[np.where(inds==i)]))

    masterx=np.array(masterx)
    mastery=np.array(mastery)
    std=np.array(std)
    
    mask=np.isfinite(masterx)
    masterx=masterx[mask][1:-1]
    mastery=mastery[mask][1:-1]
    std=std[mask][1:-1]
    
    plt.plot(masterx,mastery, label="Median Profile",c='red')
    plt.legend()
        
    return(masterx,mastery,std)
    

