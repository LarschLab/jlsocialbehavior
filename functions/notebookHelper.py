import models.experiment as xp
import functions.matrixUtilities_joh as mu
import models.experiment_set as es
import glob
import pandas as pd
from datetime import datetime


def readExperiment(csvFile, keepData=False):
    tmp = es.experiment_set(csvFile=csvFile)
    if keepData:
        return tmp
    else:
        return 1


def savedCsvToDf(txtPaths, baseDir='d:\\data\\', seachString='*siSummary*.csv'):
    csvPath = []
    for f in [mu.splitall(x)[-1][:-4] for x in txtPaths]:
        csvPath.append(glob.glob(baseDir+f+seachString)[0])

    df=pd.DataFrame()
    i=0
    for fn in csvPath:
        print(fn)
        tmp = pd.read_csv(fn, index_col=0, sep=',')
        tmp.animalSet = i
        tmp.animalIndex = tmp.animalIndex+(i*15)
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
