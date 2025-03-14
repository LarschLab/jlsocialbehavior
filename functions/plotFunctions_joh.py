import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
inToCm=2.54
import seaborn as sns
import scipy.stats as sta



def plotMapWithXYprojections(Map2D,projectWidth,targetSubplot,mapLim=31,projectionLim=0.03):
    #plot a heatmap with mean 'projections' left and bottom of heatmap
    inner_grid = gridspec.GridSpecFromSubplotSpec(3, 3,subplot_spec=targetSubplot,wspace=0.05, hspace=0.05)
    plt.subplot(inner_grid[:-1,:-1])   
    plt.imshow(Map2D,cmap="bwr",interpolation='gaussian', extent=[-mapLim,mapLim,-mapLim,mapLim],clim=(-projectionLim, projectionLim),origin='lower')
    #plt.title('accel=f(pos_n)')
    plt.ylabel('y [mm]')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',
        right='off',
        top='off',
        labelbottom='off') # labels along the bottom edge are off
    plt.plot([0, 0], [-mapLim, mapLim], 'k:', lw=1)
    plt.plot([-mapLim, mapLim], [0, 0], 'k:', lw=1)
    
    plt.ylim([-mapLim,mapLim])
    plt.xlim([-mapLim,mapLim])
    
    
    plt.subplot(inner_grid[:-1,-1])
    avgRange=np.arange(mapLim-projectWidth,mapLim+projectWidth)
    yprofile=np.nanmean(Map2D[:,avgRange],axis=1)
    x=np.arange(np.shape(yprofile)[0])-(np.ceil(np.shape(yprofile)[0])/2)
    plt.plot(yprofile,x,'b.-',markersize=2)
    plt.xlim([-projectionLim, projectionLim])
    plt.ylim([x[0],x[-1]])
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        top='off',
        left='off',
        labelleft='off') # labels along the bottom edge are off
    plt.locator_params(axis='x',nbins=4)
    plt.plot([0, 0], [-mapLim, mapLim], 'r:', lw=1)
    plt.plot([-mapLim, mapLim], [0, 0], 'r:', lw=1)
    
    plt.subplot(inner_grid[-1,:-1])
    yprofile=np.nanmean(Map2D[avgRange,:],axis=0)
    x=np.arange(np.shape(yprofile)[0])-(np.ceil(np.shape(yprofile)[0])/2)
    plt.plot(x,yprofile,'b.-',markersize=2)
    plt.xlabel('x [mm]')
    plt.ylabel('accel')
    plt.ylim([-projectionLim, projectionLim]) 
    plt.xlim([x[0],x[-1]])
    plt.locator_params(axis='y',nbins=4)
    plt.plot([0, 0], [-mapLim, mapLim], 'r:', lw=1)
    plt.plot([-mapLim, mapLim], [0, 0], 'r:', lw=1)
    return 1

def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)


def plotGroupComparison(data, groups, param, labels):
    ncols = np.array(param).shape[0]
    fig, ax = plt.subplots(1, ncols, figsize=(ncols * 3 / inToCm, 5 / inToCm))

    palSimpleBar = ['r', 'gray'][::-1]

    for i, p in enumerate(param):
        sns.pointplot(y=p,
                      x='group',
                      hue='group',
                      data=data,
                      ax=ax[i],
                      linestyle='none',
                      hue_order=groups,
                      order=groups,
                      palette='dark:k',
                      # legend=True,
                      # palette=palSimpleBar,
                      errorbar='sd')

        sns.stripplot(y=p,
                      x='group',
                      hue='group',
                      data=data,
                      ax=ax[i],
                      hue_order=groups,
                      order=groups,
                      palette=palSimpleBar,
                      alpha=.5,
                      clip_on=False,
                      zorder=-100
                      # linewidth=2,
                      )

        ax[i].set_xlabel('')
        ax[i].set_xticks([])
        ax[i].set_ylabel(labels[i])

        sns.despine()
        ax[i].legend().remove()

        d2 = data.loc[data['group'] == groups[0], p]  # iloc[:,2:]
        d1 = data.loc[data['group'] == groups[1], p]  # iloc[:,2:]

        s, p = sta.ttest_ind(d1[~np.isnan(d1)], d2[~np.isnan(d2)])

        d = cohen_d(d1[~np.isnan(d1)], d2[~np.isnan(d2)])

        if (i > -10):
            ax[i].set_ylim(bottom=0, top=None)
        else:
            ax[i].axhline(0, ls=":", color='gray')

        txtHeight = ax[i].get_ylim()[1] + (np.diff(ax[i].get_ylim()) * 0.02)
        a = "p={:01.3f}".format(p) if p >= 0.001 else "p={:01.0e}".format(p)
        ax[i].text(-0.4, txtHeight, a)
        txtHeight = ax[i].get_ylim()[1] + (np.diff(ax[i].get_ylim()) * 0.12)
        ax[i].text(-0.4, txtHeight, "d={:01.2f}".format(d))

    plt.tight_layout()

    l = ax[i].legend()
    h, l = ax[i].get_legend_handles_labels()
    ax[i].legend(h[2:], groups, frameon=False, ncol=1,
                 loc=7, title='', bbox_to_anchor=(2, 0.8),
                 labelspacing=0,
                 borderpad=0,
                 handlelength=0,
                 columnspacing=2)  # <<<<<<<< This is where the magic happens