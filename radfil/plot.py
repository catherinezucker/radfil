import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from astropy.wcs import WCS

class RadFilPlotter(object):

    '''
    A class to plot the results in radfil objects.
    '''

    def __init__(self, radobj):
        self.radobj = radobj

    def plotCuts(self, savefig = False):
        wcs = WCS(self.radobj.header)
        fig = plt.figure(figsize = (5, 5))
        ax = fig.gca(projection = wcs)

        ax.imshow(self.radobj.mask, origin='lower', cmap='binary_r', interpolation='none')
        ax.plot(self.radobj.xspline, self.radobj.yspline, 'r', label='fit', lw=2, alpha=0.25)
        if self.radobj.shift:
            ax.plot(np.asarray(self.radobj.dictionary_cuts['plot_peaks'])[:, 0],
                    np.asarray(self.radobj.dictionary_cuts['plot_peaks'])[:, 1],
                    'b.', markersize = 6.)

        for n in range(len(self.radobj.dictionary_cuts['plot_cuts'])):
            start, end = self.radobj.dictionary_cuts['plot_cuts'][n]
            ax.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth = 1.)

        ax.set_xlim(-.5, self.radobj.mask.shape[1]-.5)
        ax.set_ylim(-.5, self.radobj.mask.shape[0]-.5)



    #def plotProfile():

    #def plotBGSubtract():

    #def plotRadFit():
