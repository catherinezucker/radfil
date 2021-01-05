import numpy as np
import numbers
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from . import styles


def plotCuts(radobj, ax):

    if hasattr(radobj, 'dictionary_cuts'):
        dictionary_cuts = radobj.dictionary_cuts.copy()
    else:
        raise ValueError('Please run build_profile before plotting.')

    # plot the peaks
    if dictionary_cuts['plot_peaks'] is not None:
        toPlot = np.asarray(dictionary_cuts['plot_peaks'])

        ax.plot(toPlot[:, 0].astype(int), toPlot[:, 1].astype(int), 'b.',
                markersize = 12.,
                alpha=0.75,
                zorder = 999, markeredgecolor='white',markeredgewidth=0.5)

    # plot the cuts
    if dictionary_cuts['plot_cuts'] is not None:
        toPlot = dictionary_cuts['plot_cuts']

        [ax.plot(np.asarray(cut)[:, 0], np.asarray(cut)[:, 1], 'r-', linewidth = 1.)\
         for cut in toPlot]


    return ax


class RadFilPlotter(object):

    '''
    A class to plot the results in radfil objects.
    '''

    def __init__(self, radobj):
        self.radobj = radobj


    def plotCuts(self, ax, savefig = False):

        ## prepare
        vmin, vmax = np.nanmin(self.radobj.image[self.radobj.mask]), np.nanpercentile(self.radobj.image[self.radobj.mask], 98.)
        xmin, xmax = np.where(self.radobj.mask)[1].min(), np.where(self.radobj.mask)[1].max()
        ymin, ymax = np.where(self.radobj.mask)[0].min(), np.where(self.radobj.mask)[0].max()


        if self.radobj.cutting:


            ## plotting
            ax.imshow(self.radobj.image,
                      origin='lower',
                      cmap='gray',
                      interpolation='none',
                      norm = colors.Normalize(vmin = vmin, vmax =  vmax))
            ax.contourf(self.radobj.mask,
                        levels = [0., .5],
                        colors = 'w')
            ax.plot(self.radobj.xspline, self.radobj.yspline, 'r', label='fit', lw=3, alpha=1.0)
            ax.set_xlim(max(0., xmin-.1*(xmax-xmin)), min(self.radobj.mask.shape[1]-.5, xmax+.1*(xmax-xmin)))
            ax.set_ylim(max(0., ymin-.1*(ymax-ymin)), min(self.radobj.mask.shape[0]-.5, ymax+.1*(ymax-ymin)))

        else:

            ## plotting
            ax.imshow(self.radobj.image,
                      origin='lower',
                      cmap='gray',
                      interpolation='none',
                      norm = colors.Normalize(vmin = vmin, vmax =  vmax))
            ax.contourf(self.radobj.mask,
                        levels = [0., .5],
                        colors = 'w')
            ax.plot(line.xy[0], line.xy[1], 'r', label='fit', lw=2, alpha=0.5)
            ax.set_xlim(max(0., xmin-.1*(xmax-xmin)), min(self.radobj.mask.shape[1]-.5, xmax+.1*(xmax-xmin)))
            ax.set_ylim(max(0., ymin-.1*(ymax-ymin)), min(self.radobj.mask.shape[0]-.5, ymax+.1*(ymax-ymin)))


        plotCuts(self.radobj, ax)


    def plotFits(self, ax, plotFeature):

        if isinstance(plotFeature, str):
            if plotFeature.lower() == 'model':

                if self.radobj.bgdist is not None:
                    xplot = self.radobj.xall
                    yplot = self.radobj.yall - self.radobj.bgfit(xplot)
                    xlim=np.max(self.radobj.bgdist*1.5)


                else:
                    xplot=self.radobj.xall
                    yplot=self.radobj.yall
                    xlim=np.max(np.absolute(self.radobj.fitdist))*1.5


                ## Plot model
                #Adjust axis limit based on percentiles of data
                #axis.set_xlim(np.min(self.radobj.xall), np.max(self.radobj.xall))
                #xlim=np.max(np.absolute([np.nanpercentile(self.radobj.xall[np.isfinite(self.radobj.yall)],1),np.nanpercentile(self.radobj.xall[np.isfinite(self.radobj.yall)],99)]))

                if not self.radobj.fold:
                    ax.set_xlim(-xlim,+xlim)
                else:
                    ax.set_xlim(0., +xlim)
                ax.set_ylim(np.nanpercentile(yplot,0)-np.abs(0.5*np.nanpercentile(yplot,0)),np.nanpercentile(yplot,99.9)+np.abs(0.25*np.nanpercentile(yplot,99.9)))


                ax.plot(xplot, yplot, 'k.', markersize = 1., alpha=styles.get_scatter_alpha(len(self.radobj.xall))) 
                if self.radobj.binning:
                    if self.radobj.bgdist is not None:
                        plotbinx, plotbiny = np.ravel(list(zip(self.radobj.bins[:-1], self.radobj.bins[1:]))), np.ravel(list(zip(self.radobj.mastery-self.radobj.bgfit(self.radobj.masterx), self.radobj.mastery-self.radobj.bgfit(self.radobj.masterx))))
                    else:
                        plotbinx, plotbiny = np.ravel(list(zip(self.radobj.bins[:-1], self.radobj.bins[1:]))), np.ravel(list(zip(self.radobj.mastery, self.radobj.mastery)))
                    ax.plot(plotbinx, plotbiny,
                              'r-')

                # Plot the range
                if self.radobj.fitdist is not None:
                    ## symmetric fitting range
                    if isinstance(self.radobj.fitdist, numbers.Number):
                        ax.fill_between([-self.radobj.fitdist, self.radobj.fitdist], *ax.get_ylim(),
                                          facecolor = (0., 0., 1., .05),
                                          edgecolor = 'b',
                                          linestyle = '--',
                                          linewidth = 1.)
                    ## asymmetric fitting range
                    elif np.asarray(self.radobj.fitdist).shape == (2,):
                        plot_fitdist = self.radobj.fitdist.copy()
                        plot_fitdist[~np.isfinite(plot_fitdist)] = np.asarray(ax.get_xlim())[~np.isfinite(plot_fitdist)]
                        ax.fill_between(plot_fitdist, *ax.get_ylim(),
                                          facecolor = (0., 0., 1., .05),
                                          edgecolor = 'b',
                                          linestyle = '--',
                                          linewidth = 1.)
                ## no fitting range; all data are used
                else:
                    ax.fill_between(ax.get_xlim(), *ax.get_ylim(),
                                      facecolor = (0., 0., 1., .05),
                                      edgecolor = 'b',
                                      linestyle = '--',
                                      linewidth = 1.)

                # Plot the predicted curve
                ax.plot(np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],500), self.radobj.profilefit(np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],500)), 'b-', lw = 3., alpha = .6)


                ax.text(0.03, 0.95,"{}={:.2E}\n{}={:.2f}\n{}={:.2f}".format(self.radobj.profilefit.param_names[0],self.radobj.profilefit.parameters[0],self.radobj.profilefit.param_names[1],self.radobj.profilefit.parameters[1],self.radobj.profilefit.param_names[2],self.radobj.profilefit.parameters[2]),ha='left',va='top', fontweight='bold',fontsize=20,transform=ax.transAxes)#,bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':1.0, 'pad':1})
                ax.text(0.97, 0.95,"{}\nFit".format(self.radobj.fitfunc.capitalize()), ha='right',va='top', color='blue',fontweight='bold', fontsize=20, transform=ax.transAxes)#,bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':1.0, 'pad':1})
                #ax.tick_params(labelsize=14)


            elif plotFeature.lower() == 'bg':

                if self.radobj.bgdist is None:
                    raise ValueError('No bgfit in the radfil object. Rerun fit_profile.')

                #xlim=np.max(np.absolute([np.nanpercentile(self.radobj.xall[np.isfinite(self.radobj.yall)],1),np.nanpercentile(self.radobj.xall[np.isfinite(self.radobj.yall)],99)]))
                xlim=np.max(self.radobj.bgdist*1.5)

                if not self.radobj.fold:
                    ax.set_xlim(-xlim,+xlim)
                else:
                    ax.set_xlim(0., +xlim)
                ax.set_ylim(np.nanpercentile(self.radobj.yall,0)-np.abs(0.5*np.nanpercentile(self.radobj.yall,0)),np.nanpercentile(self.radobj.yall,99.9)+np.abs(0.25*np.nanpercentile(self.radobj.yall,99.9)))

                ax.plot(self.radobj.xall, self.radobj.yall, 'k.', markersize = 1., alpha=styles.get_scatter_alpha(len(self.radobj.xall))) 

                ##########
                if self.radobj.binning:
                    plotbinx, plotbiny = np.ravel(list(zip(self.radobj.bins[:-1], self.radobj.bins[1:]))), np.ravel(list(zip(self.radobj.mastery, self.radobj.mastery)))
                    ax.plot(plotbinx, plotbiny,
                              'r-')

                # Plot the range
                plot_bgdist = self.radobj.bgdist.copy()
                plot_bgdist[~np.isfinite(plot_bgdist)] = np.asarray(ax.get_xlim())[~np.isfinite(plot_bgdist)]
                ax.fill_between(plot_bgdist, *ax.get_ylim(),
                                  facecolor = (0., 1., 0., .05),
                                  edgecolor = 'g',
                                  linestyle = '--',
                                  linewidth = 1.)
                ax.fill_between(-plot_bgdist, *ax.get_ylim(),
                                  facecolor = (0., 1., 0., .05),
                                  edgecolor = 'g',
                                  linestyle = '--',
                                  linewidth = 1.)
                ax.plot(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 500), self.radobj.bgfit(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 500)),'g-', lw=3)
                #ax.set_xticklabels([])
                #ax.tick_params(labelsize=14)

                xplot = self.radobj.xall
                yplot = self.radobj.yall - self.radobj.bgfit(xplot)


                #Add labels#
                if self.radobj.bgfit.degree == 1:
                    ax.text(0.03, 0.95,"y=({:.2E})x+({:.2E})".format(self.radobj.bgfit.parameters[1],self.radobj.bgfit.parameters[0]),ha='left',va='top', fontweight='bold',fontsize=20, transform=ax.transAxes)#,bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':1.0, 'pad':1})
                elif self.radobj.bgfit.degree == 0:
                    ax.text(0.03, 0.95,"y=({:.2E})".format(self.radobj.bgfit.c0.value),ha='left',va='top', fontweight='bold',fontsize=20, transform=ax.transAxes)
                else:
                    warnings.warn("Labeling BG functions of higher degrees during plotting are not supported yet.")
                ax.text(0.97, 0.95,"Background\nFit", ha='right',va='top', fontweight='bold',fontsize=20, color='green',transform=ax.transAxes)#,bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':1.0, 'pad':1})

            else:
                raise ValueError('plotFeature has to be either "model" or "bg".')

        else:
            raise ValueError('plotFeature has to be either "model" or "bg".')
