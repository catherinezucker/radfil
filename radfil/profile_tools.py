import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from skimage import morphology
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


def profile_builder(radobj, point, derivative, shift = True, wrap = False):
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

    wrap: boolean
    Indicates whether to wrap around the central pixel, so that the final profile
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
    stack = sorted(list(set(zip(np.concatenate([edgex, x_edgey]),\
                       np.concatenate([y_edgex, edgey])))))
    centers =  stack[:-1]+.5*np.diff(stack, axis = 0)

    # Make the mask for pixels that the cut passes through.
    line_mask = np.zeros(image.shape, dtype = bool)
    line_mask[np.round(centers[:, 1]).astype(int), np.round(centers[:, 0]).astype(int)] = True

    # Create the final mask; find the region where the initial point is at and
    # exclude regions that are not connected. (For a very curved spine.)
    final_mask = (line_mask*mask)
    final_mask = (morphology.label(final_mask) == morphology.label(final_mask)[int(round(y0)), int(round(x0))])

    # Extract the profile from the image.
    final_idx = sorted(zip(np.where(final_mask)[1], np.where(final_mask)[0]))
    image_line = image[np.asarray(final_idx)[:, 1], np.asarray(final_idx)[:, 0]]

    # Plot.
    peak_finder = [t for t in centers if (round(t[0]), round(t[1])) in final_idx]
    start, end = peak_finder[0], peak_finder[-1]
    xpeak, ypeak = np.asarray(peak_finder)[image_line == np.max(image_line)][0]
    axis.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth = 1.)

    # Shift.
    if shift:
        #xpeak, ypeak = np.asarray(peak_finder)[image_line == np.max(image_line)][0]
        #axis.plot(xpeak, ypeak, 'b.', markersize = 6.)
        final_dist = np.asarray([np.sqrt((t[0]-xpeak)**2.+(t[1]-ypeak)**2.) for t in centers\
                  if (round(t[0]), round(t[1])) in final_idx])

    else:
        final_dist = np.asarray([np.sqrt((t[0]-x0)**2.+(t[1]-y0)**2.) for t in centers\
                      if (round(t[0]), round(t[1])) in final_idx])


    # Unwrap
    if (not wrap):
            pos0 = np.where(final_dist == np.min(final_dist))[0][0]
            if (derivative[1] > 0.):
                final_dist[:pos0] = -final_dist[:pos0]
            elif (derivative[1] < 0.):
                final_dist[(pos0+1):] = -final_dist[(pos0+1):]



    return final_dist, image_line, (xpeak, ypeak), (start, end)



### The following functions might be redundant after profile_builder gets used.
