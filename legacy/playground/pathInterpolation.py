# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 10:33:41 2017

@author: jlarsch
"""
import numpy as np
import math
import matplotlib.pyplot as plt

part=np.arange(0,10*60*30,1)
x=experiment.pair.animals[0].ts.rawTra().x()[108000:108000+10*60*30]
y=experiment.pair.animals[0].ts.rawTra().y()[108000:108000+10*60*30]

xd=np.diff(x)
yd=np.diff(y)
xd=np.hstack([0,xd])
yd=np.hstack([0,yd])
dist = np.sqrt(xd**2+yd**2)

plt.figure()
plt.hist(dist,100)


def equalizePath(x,y,precision=2):
    

    M = len(x)*100
    t = np.linspace(0, len(x), M)
    xi = np.interp(t, np.arange(len(x)), x)
    yi = np.interp(t, np.arange(len(y)), y)
    

    i, idx = 0, [0]
    while i < len(xi)-1:
        total_dist = 0
        for j in range(i+1, len(xi)):
            total_dist = math.sqrt((xi[j]-xi[i])**2 + (yi[j]-yi[i])**2)
            if total_dist > precision:
                idx.append(j)
                break
        i = j+1
    
    xn = xi[idx]
    yn = yi[idx]
    
    # Interpolate values for x and y.
    t = np.arange(len(xn))
    t2 = np.linspace(0, len(xn), len(x))
    # One-dimensional linear interpolation.
    xnn = np.interp(t2, t, xn)
    ynn = np.interp(t2, t, yn)
    return xnn,ynn

def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho
    
fig, ax = plt.subplots()
ax.plot(xnn[part],ynn[part],'go-')
ax.set_aspect('equal')
plt.show()

xnnd=np.diff(xnn)
ynnd=np.diff(ynn)
nndist = np.sqrt(xnnd**2+ynnd**2)
plt.figure()
plt.hist(nndist,100)



#np.savetxt(avi_path+".csv", np.array([xnn,ynn]).T, '%10.5f',delimiter=",")

#t=np.arange(0,10*60*30*30,30)/1000.0
t=np.linspace(0,2*np.pi,1000)
xs=((np.sin(t)+1.5*np.sin(2*t))/np.pi)*180+256
ys=((np.cos(t)-1.5*np.cos(2*t))/np.pi)*180+256
plt.figure()
plt.plot(xs,ys)


xsnn,ysnn=equalizePath(xs,ys)
th,rh=cart2pol(xsnn-256,ysnn-256)
plt.figure()
plt.hist(rh)

#t=np.arange(0,10*60*30*30,30)/1000.0
#xs=np.sin(t*(1.5/5.4))*180+256
#ys=np.cos(t*(1.5/5.4))*180+256
plt.figure()
plt.plot(xsnn,ysnn)

xsd=np.diff(xsnn)
ysd=np.diff(ysnn)
xsd=np.hstack([0,xsd])
ysd=np.hstack([0,ysd])
sdist = np.sqrt(xsd**2+ysd**2)
plt.figure()
plt.hist(sdist,100)

tscale=30*(dist*(1/5.4))/1000.0
tscaleCum=np.cumsum(tscale)

xs2=np.sin(tscaleCum)*180+256
ys2=np.cos(tscaleCum)*180+256
plt.figure()
plt.plot(xs2,ys2)
xsd2=np.diff(xs2)
ysd2=np.diff(ys2)
xsd2=np.hstack([0,xsd2])
ysd2=np.hstack([0,ysd2])
sdist2 = np.sqrt(xsd2**2+ysd2**2)
plt.figure()
plt.hist(sdist2,100)


modeList1=np.repeat('skype',len(t))
modeList2=np.repeat('fix_real_rSpeed',len(t))
modeList3=np.repeat('fix_real_cSpeed',len(t))
modeList4=np.repeat('fix_circ_cSpeed',len(t))
modeList5=np.repeat('fix_circ_rSpeed',len(t))
modeList6=np.repeat('still',len(t))

modeListAll=np.hstack([modeList1,modeList2,modeList3,modeList4,modeList5,modeList6])
xskype=np.repeat(0,len(t))
yskype=np.repeat(0,len(t))



xAll=np.hstack([xskype,x,xnn,xs,xs2,xskype])
yAll=np.hstack([yskype,y,ynn,ys,ys2,yskype])

ab = np.zeros(xAll.size, dtype=[('var1', 'S32'), ('var2', float),('var3', float)])
ab['var1'] = modeListAll
ab['var2'] = xAll
ab['var3'] = yAll

np.savetxt(avi_path+"knot.csv", ab, fmt="%s, %10.3f, %10.3f")

#np.savetxt(avi_path+".csv", np.array([modeListAll,xAll,xAll]).T, ['%s','%10.5f','%10.5f'],delimiter=",")


