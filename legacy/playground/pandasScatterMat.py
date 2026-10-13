# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 22:29:19 2016

@author: johannes
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DJF_con_SH_east_finite_all=np.random.random(1000)
DJF_con_SH_west_finite_all=np.random.random(1000)
DJF_con_SH_atl_finite_all=np.random.random(1000)
DJF_con_SH_zonal_finite_all=np.random.random(1000)

DJF_con_NH_east_finite_all=np.random.random(1000)
DJF_con_NH_west_finite_all=np.random.random(1000)
DJF_con_NH_atl_finite_all=np.random.random(1000)
DJF_con_NH_zonal_finite_all=np.random.random(1000)



df_SH = pd.DataFrame({'East_Pacific_SH':DJF_con_SH_east_finite_all, 
                   'West_Pacific_SH':DJF_con_SH_west_finite_all,
                   'Atl_SH':DJF_con_SH_atl_finite_all,
                  'Zonal_SH':DJF_con_SH_zonal_finite_all})

df_NH = pd.DataFrame({'East_Pacific_NH':DJF_con_NH_east_finite_all, 
                   'West_Pacific_NH':DJF_con_NH_west_finite_all,
                   'Atl_NH':DJF_con_NH_atl_finite_all,
                  'Zonal_NH':DJF_con_NH_zonal_finite_all})

region_name=np.array(['East_Pacific_SH', 'West_Pacific_SH', 'Atl_SH', 'Zonal_SH'])


plt.suptitle('Control Correlations')
axes = pd.scatter_matrix(df_SH, alpha=0.2, diagonal='kde')
corr = df_SH.corr().as_matrix()
for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
    axes[j, i].annotate("%.3f" %corr[j,i], (.8, .9), xycoords='axes fraction', ha='center', va='center')
plt.title('Control DJF SH', size = 15)
#plt.savefig(filename='DJF_SH_Control_Scatter.pdf', ftype='pdf')
#plt.show()

plt.subplot(212)
axes2 = pd.scatter_matrix(df_NH, alpha=0.2, diagonal='kde')
corr2 = df_NH.corr().as_matrix()
for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
    axes2[j, i].annotate("%.3f" %corr2[j,i], (.8, .9), xycoords='axes fraction', ha='center', va='center')
plt.title('Control DJF NH', size = 15)
#plt.savefig(filename='DJF_NH_Control_Scatter.pdf', ftype='pdf')
plt.show()