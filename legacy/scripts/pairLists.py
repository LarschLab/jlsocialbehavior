# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:36:26 2017

@author: jlarsch
"""

import numpy as np


#each with next
a=np.zeros((15,15))
np.fill_diagonal(a,1)
a=np.roll(a,1,axis=1)
b=a.tolist()
path="E:\\00_bonsai_ffmpeg_out\\pairLists\\PL_15_eachWithNext.txt"
np.savetxt(path,a,fmt='%i')

#each with each except self
a=np.ones((16,16))
np.fill_diagonal(a,0)
b=a.tolist()
path="E:\\00_bonsai_ffmpeg_out\\pairLists\\PL_15_eachWithEachExceptSelf.txt"
np.savetxt(path,a,fmt='%i')

#pairs
a=np.zeros((15,15))
np.fill_diagonal(a,1)
a=np.roll(a,1,axis=1)
a[1::2,:]=0
a.tolist()
a2=np.zeros((15,15))
np.fill_diagonal(a2,1)
a2=np.roll(a2,-1,axis=1)
a2[0::2,:]=0
b=(a+a2).tolist()
b[-1][0]=0.
b[-1][-1]=1.
path="E:\\00_bonsai_ffmpeg_out\\pairLists\\PL_15_pairsLastSelf.txt"
np.savetxt(path,b,fmt='%i')

#each with last
a=np.zeros((16,16))
a[-1,:-1]=1
b=a.tolist()
path="E:\\00_bonsai_ffmpeg_out\\pairLists\\PL_16_eachWithLast.txt"
np.savetxt(path,a,fmt='%i')


#each with self
a=np.zeros((16,16))
np.fill_diagonal(a,1)
a[-1,-1]=0
b=a.tolist()
path="E:\\00_bonsai_ffmpeg_out\\pairLists\\PL_16_eachWithSelf.txt"
np.savetxt(path,a,fmt='%i')