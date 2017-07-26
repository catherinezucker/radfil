import numpy as np
import numbers
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

    def plotFits(self, savefig = False):
        fig, ax = plt.subplots(figsize = (7., 5.), ncols = 2, nrows = 2)
        ## plummer(+bgfit, if bgfit)
        axis = ax[0, 0]
        axis.plot(self.radobj.xall, self.radobj.yall, 'k.', markersize = 8., alpha = .05)
        ## Plot bins if binned.
        if self.radobj.binning:
            stepx, stepy = np.zeros(len(self.radobj.masterx)*2)*np.nan, np.zeros(len(self.radobj.mastery)*2) * np.nan
            stepx[::2] = self.radobj.masterx-.5*np.diff(self.radobj.masterx)[0]  ## assuming linear binning
            stepx[1::2] = self.radobj.masterx+.5*np.diff(self.radobj.masterx)[0]
            stepy[::2], stepy[1::2] = self.radobj.mastery, self.radobj.mastery
            axis.plot(stepx, stepy, 'k-', alpha = .4)
        ## Plot bg if bg removed.
        if isinstance(self.radobj.bgdist, numbers.Number):
            if self.radobj.wrap:
                axis.plot(self.radobj.xbg, self.radobj.ybg, 'g.', markersize = 8., alpha = .15)
                axis.plot(self.radobj.xfit, self.radobj.yfit+self.radobj.bgfit, 'b.', markersize = 8., alpha = .15)
                ## Plot the fits (profilefit + bgfit)
                xplot = np.linspace(np.min(self.radobj.xall), np.max(self.radobj.xall), 100)
                axis.plot(xplot, self.radobj.bgfit+self.radobj.profilefit_plummer(xplot), 'b-', lw = 3., alpha = .6)
            else:
                axis.plot(self.radobj.xbg, self.radobj.ybg, 'g.', markersize = 8., alpha = .15)
                axis.plot(self.radobj.xfit, self.radobj.yfit+self.radobj.bgfit(self.radobj.xfit), 'b.', markersize = 8., alpha = .15)
                ## Plot the fits (profilefit + bgfit)
                xplot = np.linspace(np.min(self.radobj.xall), np.max(self.radobj.xall), 100)
                axis.plot(xplot, self.radobj.bgfit(xplot)+self.radobj.profilefit_plummer(xplot), 'b-', lw = 3., alpha = .6)


        else:
            axis.plot(self.radobj.xfit, self.radobj.yfit, 'b.', markersize = 8., alpha = .15)
            ## Plot the fits (profilefit)
            xplot = np.linspace(np.min(self.radobj.xall), np.max(self.radobj.xall), 100)
            axis.plot(xplot, self.radobj.profilefit_plummer(xplot), 'b-', lw = 3., alpha = .6)
        ## Adjust the plot
        axis.set_xlim(np.min(self.radobj.xall), np.max(self.radobj.xall))
        #axis.set_yscale('log')
        axis.set_xticklabels([])

        ## plummer residual
        axis = ax[0, 1]
        axis.plot(self.radobj.xfit, self.radobj.yfit-self.radobj.profilefit_plummer(self.radobj.xfit), 'b.', markersize = 8., alpha = .3)
        ## Adjust the plot
        axis.set_xlim(np.min(self.radobj.xall), np.max(self.radobj.xall))
        axis.set_xticklabels([])
        axis.set_yticklabels([])

        ## gaussian(+bgfit, if bgfit)
        axis = ax[1, 0]
        axis.plot(self.radobj.xall, self.radobj.yall, 'k.', markersize = 8., alpha = .15)
        ## Plot bins if binned.
        if self.radobj.binning:
            stepx, stepy = np.zeros(len(self.radobj.masterx)*2)*np.nan, np.zeros(len(self.radobj.mastery)*2) * np.nan
            stepx[::2] = self.radobj.masterx-.5*np.diff(self.radobj.masterx)[0]  ## assuming linear binning
            stepx[1::2] = self.radobj.masterx+.5*np.diff(self.radobj.masterx)[0]
            stepy[::2], stepy[1::2] = self.radobj.mastery, self.radobj.mastery
            axis.plot(stepx, stepy, 'k-', alpha = .4)
        ## Plot bg if bg removed.
        if isinstance(self.radobj.bgdist, numbers.Number):
            if self.radobj.wrap:
                axis.plot(self.radobj.xbg, self.radobj.ybg, 'g.', markersize = 8., alpha = .15)
                axis.plot(self.radobj.xfit, self.radobj.yfit+self.radobj.bgfit, 'r.', markersize = 8., alpha = .15)
                ## Plot the fits (profilefit + bgfit)
                xplot = np.linspace(np.min(self.radobj.xall), np.max(self.radobj.xall), 100)
                axis.plot(xplot, self.radobj.bgfit+self.radobj.profilefit_gaussian(xplot), 'r-', lw = 3., alpha = .6)
            else:
                axis.plot(self.radobj.xbg, self.radobj.ybg, 'g.', markersize = 8., alpha = .15)
                axis.plot(self.radobj.xfit, self.radobj.yfit+self.radobj.bgfit(self.radobj.xfit), 'r.', markersize = 8., alpha = .15)
                ## Plot the fits (profilefit + bgfit)
                xplot = np.linspace(np.min(self.radobj.xall), np.max(self.radobj.xall), 100)
                axis.plot(xplot, self.radobj.bgfit(xplot)+self.radobj.profilefit_gaussian(xplot), 'r-', lw = 3., alpha = .6)

        else:
            axis.plot(self.radobj.xfit, self.radobj.yfit, 'r.', markersize = 8., alpha = .15)
            ## Plot the fits (profilefit)
            xplot = np.linspace(np.min(self.radobj.xall), np.max(self.radobj.xall), 100)
            axis.plot(xplot, self.radobj.profilefit_gaussian(xplot), 'r-', lw = 3., alpha = .6)
        ## Adjust the plot
        axis.set_xlim(np.min(self.radobj.xall), np.max(self.radobj.xall))
        #axis.set_yscale('log')

        ## gaussian residual
        axis = ax[1, 1]
        axis.plot(self.radobj.xfit, self.radobj.yfit-self.radobj.profilefit_gaussian(self.radobj.xfit), 'r.', markersize = 8., alpha = .3)
        ## Adjust the plot
        axis.set_xlim(np.min(self.radobj.xall), np.max(self.radobj.xall))
        axis.set_yticklabels([])
