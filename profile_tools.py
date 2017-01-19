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
    pts=np.vstack((x,y))

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
        p = paths[i]   # order of nodes
        ordered = pts[p]   # ordered nodes
        
        # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
        cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
        if cost < mindist:
            mindist = cost
            minidx = i
        
    opt_order = paths[minidx]

    xx = x[opt_order]
    yy = y[opt_order]
    
    return(xx,yy)
    
    
def maskbounds(points,radobj,ax,plot_width=True):

    """
    Determine where the perpendicular cuts touch the mask on either side of the spine. These define
    each of the "local width lines" perpendicular to the smoothed filament spine 
    
    Parameters:
    points: numpy.ndarray
       A 1d array specifying the x coordinates of the pixels at which you want to sample the widths
       
    radobj:
       An instance of the radfil_class which now contain the following relevant attributes: 
    
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
            The first derivative of radobj.xfit along the smoothed spine
    
        yprime: numpy.ndarray
            The first derivative of radobj.yfit along the smoothed spine
        
    ax: matplotlib.axes._subplots.AxesSubplot
        The axes for the current figure
        
    plot_width: boolean
        Do you want to plot each of the local width lines?
       
    Returns:
    
    leftx: array containing x coordinates at the LHS of each width line
    rightx: array containing the x coordinates at the RHS of each width line
    lefty: array containing the y coordinates on the LHS of each width line
    righty: array containing the y coordinates on the RHS of each width line
    distpc: array containing the width of each line
    finalpts: array of the final tangent points defining each width line.    
    """ 
    
    dist=[]
    distpc=[]
    finalpts=[]
    leftx=[]
    rightx=[]
    lefty=[]
    righty=[]
            
    for i in range(0,len(points)):
    
        #Determine the parameters defining the equation of the line perpendicular to the tangent
        a=points[i]
        argmin=np.argmin(np.abs(radobj.xfit-np.ones((radobj.xfit.shape[0]))*a))
        fa=radobj.yfit[argmin]   
        fprime=radobj.yprime[argmin]/radobj.xprime[argmin]
        m=-1.0/fprime
        deltay=.1
                
        deltax=(deltay/np.absolute(m))
        linerange=np.arange(1,10000,1)
                    
        #Iteratively increase the length of the width line outwards from the tangent point until you reach both sides of filament radobj.mask
        for line in linerange:
        
            findxbound=np.linspace(a-line*deltax,a+line*deltax,100)
            findybound=np.array(fa+(-1.0/fprime)*(findxbound-a)) # tangent line
            
            #The program will fail if you have a slope ---> infinity or 0. Discard these points!
            if (np.min(findxbound)<0) or (np.max(findxbound)>radobj.image.shape[1]-1) or (np.min(findybound)<0) or (np.max(findybound)>radobj.image.shape[0]-1):
                print(a,'Bad point...skipping')
                break
            
            boundarr=radobj.mask[findybound.astype(int),findxbound.astype(int)]
            halved=findxbound.shape[0]/2

            #Is the radobj.mask entirely confined to either LHS or RHS of the spine?
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
                distpc.append(dist*radobj.radobj.imagescale)
                
                if plot_width==True:
                    ax.plot(fin_x,fin_tanperp,ls='solid',c='r',alpha=0.3,zorder=2)  
                    
                finalpts.append(a)
                       
                break
                
            #Or is there radobj.mask on both sides of the tangent point?
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
                distpc.append(dist*radobj.radobj.imagescale)
                    
                if plot_width==True:
                    ax.plot(fin_x,fin_tanperp,ls='solid',c='r',alpha=0.3,zorder=2)   
                    
                finalpts.append(a)
                    
                break 

    return (np.array(leftx),np.array(rightx),np.array(lefty),np.array(righty),np.array(distpc),np.array(finalpts))
    
def max_intensity(points,radobj,leftx,rightx,lefty,righty,ax):

 """
    Determine the pixel along each of the local width lines with the maximum column density value
    
    Parameters:
    finalpts: list of the final tangent points defining each width line.    

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

    for i in range(0,len(points)):
    
        #Determine the parameters defining the equation of the line perpendicular to the tangent
        a=points[i]
        argmin=np.argmin(np.abs(radobj.xfit-np.ones((radobj.xfit.shape[0]))*a))
        fa=radobj.yfit[argmin]   
        fprime=radobj.yprime[argmin]/radobj.xprime[argmin]
        m=-1.0/fprime
                
        #Sample the line along the entire local width
        findxbound=np.linspace(np.min([leftx[i],rightx[i]]),np.max([leftx[i],rightx[i]]),100)
        findybound=np.array(fa+(-1.0/fprime)*(findxbound-a)) # tangent line
               
        #Determine unique pixel values along the local width
        unique,indices,inverse,counts=np.unique(radobj.image[findybound.astype(int),findxbound.astype(int)],return_index=True,return_inverse=True,return_counts=True)
    
        uniquex=findxbound.astype(int)[indices]
        uniquey=findybound.astype(int)[indices]
    
        order=np.argsort(indices)
        indices=indices[order]
        counts=counts[order]
        
        #if same pixel value is sampled multiple times, extract only one of each instance
        extract=[]
        for j in range(0,len(counts)):
            extract.append(np.sum(counts[0:j])+counts[j]/2)
        
        coldensity=radobj.image[findybound.astype(int)[extract],findxbound.astype(int)[extract]]
    
        #of the unique pixels, determine one with maximum column density value
        argmax=np.argmax(coldensity)
        maxcolx.append(int(findxbound[extract[argmax]]))
        maxcoly.append(int(findybound[extract[argmax]]))
        
    return np.array(maxcolx),np.array(maxcoly)
    
def get_radial_prof(points,maxcolx,maxcoly,cutdist=3.0,ax,plot_max=True,plot_all=True):

 """
    Return the radial profile along each perpendicular cut across the spine, shifted to the peak column density along each local width line
    
    Parameters:
    finalpts: list of the final tangent points defining each width line.    

    radobj:
       An instance of the radfil_class which contains all the relevant attributes described in maskbounds function above
       
    maxcolx: numpy.ndarray
        1D array containing x coordinates of the pixels with the maximum column density along each local width line
        
    maxcoly: numpy.ndarray
        1D array containing y coordinates of the pixels with the maximum column density along each local width line
        
    cutdist: float
        The maximum radial distance out to which you want to sample the profile. Must be in pc. 
        
    ax: matplotlib.axes._subplots.AxesSubplot
        The axes for the current figure
        
    plot_max: boolean
        Do you want to mark the pixel with the maximum column density along each local width line? 
    
    plot_all: boolean
        Do you want to plot the points at which you're sampling your radial column density profile? 
        
    Returns: 
    xaxis: numpy.ndarray
        1D array containing the radial distances from the peak column density pixel (defined to be r=0)
    
    yaxis: numpy.ndarray
        1D array containing the column densities corresponding to the radial distances stored in xaxis 
        
 """
    
    xaxis=[]
    yaxis=[]
    
    indexedarr=np.cumsum(np.ones((img.shape))).reshape((img.shape))
    
    for i in range(0,len(points)):
        a=points[i]
        argmin=np.argmin(np.abs(radobj.xfit-np.ones((radobj.xfit.shape[0]))*a))
        fa=yfit[argmin]   
        fprime=yprime[argmin]/xprime[argmin]
        m=-1.0/fprime
        
        deltaylower=fa-3
        deltayupper=radobj.image.shape[0]-fa-3
        deltaxlower=(deltaylower/np.absolute(m))
        deltaxupper=(deltayupper/np.absolute(m))
        
        if np.sign(m)==-1:
            findxbound=np.linspace(np.clip(a-deltaxupper,0,a),np.clip(a+deltaxlower,a,radobj.image.shape[1]-3),100000)
        else:
            findxbound=np.linspace(np.clip(a-deltaxlower,0,a),np.clip(a+deltaxupper,a,radobj.image.shape[1]-3),100000)
            
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

        radialdist=[]
        center = np.where(((findxbound.astype(int)[extract]==maxcolx[i]) & (findybound.astype(int)[extract]==maxcoly[i]))==True)[0]
        centerx=findxbound[extract][center]
        centery=findybound[extract][center]
        for pt in zip(findxbound[extract],findybound[extract]):
            radialdist.append(math.hypot(pt[0] - centerx, pt[1] - centery))

        radialdist=np.array(radialdist)*radobj.imgscale
 
        leftspine=np.where(findxbound[extract]<centerx)
        radialdist[leftspine[0]]=radialdist[leftspine[0]]*-1

        deletepts=np.where(np.abs(radialdist)>cutdist)
        radialdist=np.delete(radialdist,deletepts)
        coldensity=np.delete(coldensity,deletepts)
        extract=np.delete(extract,deletepts)

        xaxis.append(radialdist)
        yaxis.append(coldensity)

        if indexedarr[findybound.astype(int)[extract],findxbound.astype(int)[extract]].shape[0]!=np.unique(indexedarr[findybound.astype(int)[extract],findxbound.astype(int)[extract]]).shape[0]:
            print(a,"Something is wrong!!! Profile has repeat column density values")

        if plot_all==True:
            ax.scatter(findxbound[extract],findybound[extract],c='green',edgecolor="None",zorder=4,s=20)
        if plot_max==True:
            ax.scatter(centerx,centery,c='blue',edgecolor="None",zorder=4,s=20,marker='o',alpha=0.3)

    return np.array(xaxis),np.array(yaxis)
    