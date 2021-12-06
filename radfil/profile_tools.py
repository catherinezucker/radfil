import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from skimage import morphology
import scipy
import matplotlib.pyplot as plt
import math
from scipy.interpolate import RegularGridInterpolator 


from scipy.spatial import distance


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

    # make the point list (N x 2)
    pts=np.vstack((x,y)).T

    # initiate the NN2 and the network graph
    try:
        clf = NearestNeighbors(2).fit(pts)
    except:
        clf = NearestNeighbors(n_neighbors = 2).fit(pts)    
    G = clf.kneighbors_graph()
    T = nx.from_scipy_sparse_matrix(G)

    # candidate paths based on the network graph
    paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(pts))]
    ## find the path with the lowest cost (distance) among the candidates
    minidx = np.argmin([np.sum(np.diagonal(distance.cdist(pts[path], pts[path]), offset = 1)) for path in paths])
    opt_order = paths[minidx]
    ## permute the head and the tail to find the correct order
    ### permutation
    opt_order_swaphead = list(opt_order)
    opt_order_swaphead[0], opt_order_swaphead[1] = opt_order_swaphead[1], opt_order_swaphead[0]
    opt_order_swaptail = list(opt_order)
    opt_order_swaptail[-1], opt_order_swaptail[-2] = opt_order_swaptail[-2], opt_order_swaptail[-1]
    ### find the correct order among the original and the two permuted
    paths_opt = [opt_order, opt_order_swaphead, opt_order_swaptail]
    minidx_opt = np.argmin([np.sum(np.diagonal(distance.cdist(pts[path], pts[path]), offset = 1)) for path in paths_opt])
    opt_order_final = paths_opt[minidx_opt]

    # return the ordered coordinates
    xx = x[opt_order_final]
    yy = y[opt_order_final]
    ## make it always go in the increasing y direction
    if yy[-1] < yy[0]:
        yy = yy[::-1]
        xx = xx[::-1]

    return(xx,yy)


def profile_builder(radobj, point, derivative, shift = True, fold = False):
    '''
    Build the profile using array manipulation, instead of looping.

    Parameters:

    radobj:
    The object containing the image, the mask, and the axis for plotting.

    point: tuple-like
    The x and y pixel coordinates (corresponding to the 1st and the 0th axes) of
    the point at the center.

    derivative: tuple-like
    Thee x and y components of the derivative of the spline at `point`.  Used to
    derive the profile cut.

    shift: boolean
    Indicates whether to shift the profile to center at the peak value.

    fold: boolean
    Indicates whether to fold around the central pixel, so that the final profile
    will be a "half profile" with the peak near/at the center (depending on
    whether it's shifted).


    Returns:

    final_dist: 1D numpy.ndarray
    The distance array.

    image_line: 1D numpy.ndarray
    The value array.

    '''
    # Read the image and the mask
    image, mask = radobj.image, radobj.mask
    # Read the plotting axis
    axis = radobj.ax

    # Read the point and double check whether it's inside the mask.
    x0, y0 = point
    if (not mask[int(round(y0)), int(round(x0))]):
        raise ValueError("The point is not in the mask.")

    # Create the grid to calculate where the profile cut crosses edges of the
    # pixels.
    shapex, shapey = image.shape[1], image.shape[0]
    edgex, edgey = np.arange(.5, shapex-.5, 1.), np.arange(.5, shapey-.5, 1.)

    # Extreme cases when the derivative is (1, 0) or (0, 1)
    if (derivative[0] == 0) or (derivative[1] == 0):
        if (derivative[0] == 0) and (derivative[1] == 0):
            raise ValueError("Both components of the derivative are zero; unable to derive a tangent.")
        elif (derivative[0] == 0):
            y_edgex = []
            edgex = []
            x_edgey = np.ones(len(edgey))*x0
        elif (derivative[1] == 0):
            y_edgex = np.ones(len(edgex))*y0
            x_edgey = []
            edgey = []

    ## The regular cases go here: calculate the crossing points of the cut and the grid.
    else:
        slope = -1./(derivative[1]/derivative[0])
        y_edgex = slope*(edgex - x0) + y0
        x_edgey = (edgey - y0)/slope + x0

        ### Mask out points outside the image.
        pts_maskx = ((np.round(x_edgey) >= 0.) & (np.round(x_edgey) < shapex))
        pts_masky = ((np.round(y_edgex) >= 0.) & (np.round(y_edgex) < shapey))

        edgex, edgey = edgex[pts_masky], edgey[pts_maskx]
        y_edgex, x_edgey = y_edgex[pts_masky], x_edgey[pts_maskx]


    # Sort the points to find the center of each segment inside a single pixel.
    ## This also deals with when the cut crosses at the 4-corner point(s).
    ## The sorting is done by sorting the x coordinates
    stack = sorted(list(set(zip(np.concatenate([edgex, x_edgey]),\
                       np.concatenate([y_edgex, edgey])))))
    centers =  stack[:-1]+.5*np.diff(stack, axis = 0)

    ## extract the values from the image and the original mask
    #setup interpolation 
    xgrid=np.arange(0.5,radobj.image.shape[1]+0.5,1.0)
    ygrid=np.arange(0.5,radobj.image.shape[0]+0.5,1.0)
    interpolator = RegularGridInterpolator((xgrid,ygrid),radobj.image.T,bounds_error=False,fill_value=None)
    
    image_line=interpolator(centers)
    #image_line = image[np.round(centers[:, 1]).astype(int), np.round(centers[:, 0]).astype(int)]
    
    mask_line = mask[np.round(centers[:, 1]).astype(int), np.round(centers[:, 0]).astype(int)]
    #### select the part of the mask that includes the original point
    mask_p0 = (np.round(centers[:, 0]).astype(int) == int(round(x0)))&\
              (np.round(centers[:, 1]).astype(int) == int(round(y0)))
    mask_line = (morphology.label(mask_line) == morphology.label(mask_line)[mask_p0])

    # Extract the profile from the image.
    ## for the points within the original mask; to find the peak
    if derivative[1] < 0.:
        image_line0 = image_line[mask_line][::-1]
        centers = centers[::-1]
        mask_line = mask_line[::-1]
        mask_p0 = mask_p0[::-1]
    else:
        image_line0 = image_line[mask_line]
    ## for the entire map
    if derivative[1] < 0.:
        image_line = image_line[::-1]
    else:
        image_line = image_line

    # Plot.
    peak_finder = centers[mask_line]
    ## find the end points of the cuts (within the original mask)
    start, end = peak_finder[0], peak_finder[-1]
    ## find the peak here
    xpeak, ypeak = peak_finder[image_line0 == np.nanmax(image_line0)][0]
    ## the peak mask is used to determine where to unfold when shift = True
    mask_peak = (np.round(centers[:, 0]).astype(int) == int(round(xpeak)))&\
                (np.round(centers[:, 1]).astype(int) == int(round(ypeak)))
    ## plot the cut
    axis.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth = 1.,alpha=1)

    # Shift.
    if shift:
        final_dist = np.hypot(centers[:, 0]-xpeak, centers[:, 1]-ypeak)
        # unfold
        pos0 = np.where(mask_peak)[0][0]
        final_dist[:pos0] = -final_dist[:pos0]
    else:
        final_dist = np.hypot(centers[:, 0]-x0, centers[:, 1]-y0)
        # unfold
        pos0 = np.where(mask_p0)[0][0]
        final_dist[:pos0] = -final_dist[:pos0]


    # Fold
    if fold:
            final_dist = abs(final_dist)

    return final_dist, image_line, (xpeak, ypeak), (start, end)
