# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 21:24:31 2016

@author: johannes
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_excel('c:/test/eg.xlsx')

standx=df['a']

def gaussian(x, a, mean, sigma):
    return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))

f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
bins = np.histogram(standx, bins = 100)[1]

from scipy.optimize import curve_fit
from scipy import stats




num_1, bins_1  = np.histogram(standx, bins = 100)
plt.hist(standx, bins = 100) #jl
plt.plot(bins_1[:-1],num_1) #jl

bins_log=np.log10(bins_1[:-1])
plt.figure()
plt.plot(bins_log,num_1)

bins_01 = np.logspace( np.log10( standx.min()  ), np.log10(standx.max() ), 100 )
x_fit = np.linspace(bins_01[0], bins_01[-1], 100)

#popt, pcov = curve_fit(gaussian, x_fit, num_1, p0=[1, np.mean(standx), np.std(standx)])
popt, pcov = curve_fit(gaussian, bins_log, num_1, p0=[1, np.mean(standx), np.std(standx)])

#y_fit = gaussian(bins_01, *popt)
y_fit = gaussian(bins_log, *popt)
plt.plot(bins_log,y_fit,'r.-')

counts, edges, patches = ax1.hist(standx, bins_01, facecolor='blue', alpha=0.5) # bins=100
area = sum(np.diff(edges)*counts)

# calculate length of each bin (required for scaling PDF to histogram)
bins_log_len = np.zeros( x_fit.size )
for ii in range( counts.size):
    bins_log_len[ii] = edges[ii+1]-edges[ii]

# Create an array of length num_bins containing the center of each bin.
centers = 0.5*(edges[:-1] + edges[1:])
# Make a fit to the samples.
shape, loc, scale = stats.lognorm.fit(standx, floc=0)
# get pdf-values for same intervals as histogram
samples_fit_log = stats.lognorm.pdf( bins_01, shape, loc=loc, scale=scale )
# oplot fitted and scaled PDF into histogram


pdf_from_fit=stats.norm.pdf(bins_log, popt[1], popt[2])

plt.plot(bins_log,pdf_from_fit*13,'g.-')

plt.plot(bins_log,np.multiply(samples_fit_log*525,'k:'))



plt.figure()
plt.hist(standx, bins_01, facecolor='blue', alpha=0.5) # bins=100

new_x = np.linspace(np.min(standx), np.max(standx), 100)
pdf = stats.norm.pdf(new_x, loc=np.log(scale), scale=shape)

plt.plot(bins_01, np.multiply(samples_fit_log,    bins_log_len)*sum(counts), 'g--', label='PDF using histogram bins', linewidth=2 )
log_adjusted_pdf=np.multiply(bins_log_len,stats.norm.pdf(bins_log, popt[1], popt[2]))
scale_factor=len(standx)/sum(log_adjusted_pdf)
plt.plot(bins_1[:-1], scale_factor*log_adjusted_pdf,'r--',linewidth=2,label='Fit: $\mu$=%.3f , $\sigma$=%.3f'%(popt[1],popt[2]) )
plt.set_xscale('log')
plt.legend(loc='best', frameon=False, prop={'size':15})

# And similar for the ax2, ax3 plots





def gaussian(x, a, mean, sigma):
    return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))

f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
bins = np.histogram(standx, bins = 100)[1]

from scipy.optimize import curve_fit
from scipy import stats


num_1, bins_1  = np.histogram(standx, np.histogram(standx, bins = 100)[1])

#log transform the bins!
bins_log=np.log10(bins_1[:-1])
bins_01 = np.logspace( np.log10( standx.min()  ), np.log10(standx.max() ), 100 )
x_fit = np.linspace(bins_01[0], bins_01[-1], 100)

#popt, pcov = curve_fit(gaussian, x_fit, num_1, p0=[1, np.mean(standx), np.std(standx)])
popt, pcov = curve_fit(gaussian, bins_log, num_1, p0=[1, np.mean(standx), np.std(standx)])

#y_fit = gaussian(bins_01, *popt)
y_fit = gaussian(bins_log, *popt)
counts, edges, patches = ax1.hist(standx, bins_01, facecolor='blue', alpha=0.5) # bins=100
area = sum(np.diff(edges)*counts)

# calculate length of each bin (required for scaling PDF to histogram)
bins_log_len = np.zeros( x_fit.size )
for ii in range( counts.size):
    bins_log_len[ii] = edges[ii+1]-edges[ii]

# Create an array of length num_bins containing the center of each bin.
centers = 0.5*(edges[:-1] + edges[1:])
# Make a fit to the samples.
shape, loc, scale = stats.lognorm.fit(standx, floc=0)
# get pdf-values for same intervals as histogram
samples_fit_log = stats.lognorm.pdf( bins_01, shape, loc=loc, scale=scale )
# oplot fitted and scaled PDF into histogram


new_x = np.linspace(np.min(standx), np.max(standx), 100)
pdf = stats.norm.pdf(new_x, loc=np.log(scale), scale=shape)
ax1.plot(new_x, pdf*sum(counts), 'k-')
ax1.plot(bins_01, np.multiply(samples_fit_log,    bins_log_len)*sum(counts), 'g--', label='PDF using histogram bins', linewidth=2 )

#ax1.plot(x_fit, stats.norm.pdf(x_fit, popt[1], popt[2])*area,'r--',linewidth=2,label='Fit: $\mu$=%.3f , $\sigma$=%.3f'%(popt[1],popt[2]) )

log_adjusted_pdf=np.multiply(bins_log_len,stats.norm.pdf(bins_log, popt[1], popt[2]))
scale_factor=len(standx)/sum(log_adjusted_pdf)
ax1.plot(bins_1[:-1], scale_factor*log_adjusted_pdf,'r--',linewidth=2,label='Fit: $\mu$=%.3f , $\sigma$=%.3f'%(popt[1],popt[2]) )

#ax1.plot(bins_1[1:], stats.norm.pdf(bins_log, popt[1], popt[2])*18,'r--',linewidth=2,label='Fit: $\mu$=%.3f , $\sigma$=%.3f'%(popt[1],popt[2]) )
ax1.set_xscale('log')
ax1.legend(loc='best', frameon=False, prop={'size':15})

# And similar for the ax2, ax3 plots