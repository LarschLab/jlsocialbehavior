import models.experiment as xp
import functions.matrixUtilities_joh as mu
import models.experiment_set as es
import glob
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt



def readExperiment(csvFile, keepData=False):
    tmp = es.experiment_set(csvFile=csvFile)
    if keepData:
        return tmp
    else:
        return 1


def savedCsvToDf(txtPaths, baseDir='d:\\data\\', seachString='*siSummary*.csv',noOfAnimals=15):
    csvPath = []
    for f in [mu.splitall(x)[-1][:-4] for x in txtPaths]:
        csvPath.append(glob.glob(baseDir+f+seachString)[0])

    df=pd.DataFrame()
    i=0
    for fn in csvPath:
        print(fn)
        tmp = pd.read_csv(fn, index_col=0, sep=',')
        tmp.animalSet = i
        tmp.animalIndex = tmp.animalIndex+(i*noOfAnimals)
        df = pd.concat([df, tmp])
        i += 1
    return df


def computeExpTimeOfDay(df):
    d = df.time
    r = datetime(int(df.time.iloc[0][:4]), 1, 1)
    t2 = [pd.to_datetime(x).replace(day=1, month=1)for x in df.time]
    t3 = [(x-r)/pd.Timedelta('1 hour') for x in t2]
    df['t2'] = t2
    df['t3'] = t3
    return df

from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def speedFft(data):
    tmp=data-np.nanmean(data)
    tmp=tmp[~np.isnan(tmp)]
    freqs = np.fft.fftfreq(tmp.size,1)
    idx = np.argsort(freqs)
    ps = np.abs(np.fft.fft(tmp))**2
    return freqs[idx],ps[idx]