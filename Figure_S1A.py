import pandas as pd
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import cm
import matplotlib.colors as colors
from colorsys import hsv_to_rgb
#import plotly.plotly as py  # tools to communicate with Plotly's server
import csv
import scipy.stats as stats
from matplotlib import pylab

def linreg(X, Y):
    """
        Summary
        Linear regression of y = ax + b
        Usage
        real, real, real = linreg(list, list)
        Returns coefficients to the regression line "y=ax+b" from x[] and y[], and R^2 Value
        """
    if len(X) != len(Y):  raise ValueError("unequal length")
    N = len(X)
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for x, y in zip(X, Y):
        Sx = Sx + x
        Sy = Sy + y
        Sxx = Sxx + x*x
        Syy = Syy + y*y
        Sxy = Sxy + x*y
    det = Sxx * N - Sx * Sx
    a, b = (Sxy * N - Sy * Sx)/det, (Sxx * Sy - Sx * Sxy)/det
    meanerror = residual = 0.0
    for x, y in zip(X, Y):
        meanerror = meanerror + (y - Sy/N)**2
        residual = residual + (y - a * x - b)**2
    RR = 1 - residual/meanerror
    ss = residual / (N-2)
    Var_a, Var_b = ss * N / det, ss * Sxx / det
    #print sqrt(Var_a),sqrt(Var_b)
    #print "y=ax+b"
    #print "N= %d" % N
    #print "a= %g \pm t_{%d;\alpha/2} %g" % (a, N-2, sqrt(Var_a))
    #print "b= %g \pm t_{%d;\alpha/2} %g" % (b, N-2, sqrt(Var_b))
    #print "R^2= %g" % RR
    #print "s^2= %g" % ss
    return a, b, RR, Var_a, Var_b

norm = colors.Normalize(vmin=1, vmax=2*len(range(2000,2015)))
sm = cm.ScalarMappable(norm, cmap=cm.Paired)
mk='o'
cnt=0

avx=[]
avy=[]

for ii in range(2,18):

    ff=open('GDP_MSAs_current_dollars.csv', 'r')
    reader=csv.reader(ff,delimiter=',')
    nGDP=[]
    
    for row in reader:
        if (row[0].isdigit() and int(row[0])>10000):
            #print(row[0],row[1],row[ii])
            nGDP.append(float(row[ii]))
    ff.close()
    
    f=open('real_GDP.csv', 'r')
    reader=csv.reader(f,delimiter=',')
    GDP=[]
    fips=[]

    for row in reader:
        if (row[0].isdigit() and int(row[0])>10000):
        #print(row[0],row[1],row[ii])
            fips.append(int(row[0]))
            GDP.append(float(row[ii]))


    fpc=open('real_GDP_pc.csv', 'r')
    readerpc=csv.reader(fpc,delimiter=',')

    GDPpc=[]
    fipspc=[]

    for row in readerpc:
        if (row[0].isdigit() and int(row[0])>10000):
        #print(row[0],row[1],row[ii])
            fipspc.append(int(row[0]))
            GDPpc.append(float(row[ii]))

    pop=[]
    for i in range(len(GDP)):
        if (fips[i]==fipspc[i]):
            pop.append(GDP[i]/GDPpc[i]*10**6)
        else:
            print (fips[i])

    xx=np.log10(pop)
    yy=np.log10(nGDP)
    av_xx=np.mean(xx)
    av_yy=np.mean(yy)
    avx.append(av_xx)
    avy.append(av_yy)

    edge_color, color = sm.to_rgba(cnt), sm.to_rgba(cnt+1)
    edge_color='white'
    cnt += 2
    plt.plot(xx,yy,marker='o',ms=10,ls='None',c=color,markeredgecolor=edge_color,markeredgewidth=.5,alpha=0.3)

    gradient, intercept, r_value, var_gr, var_it = linreg(xx,yy)
    print( "Gradient=", gradient, ", 95 % CI = [",gradient- 2.*np.sqrt(var_gr),",",gradient+2.*np.sqrt(var_gr),"]")
    print( "intercept=", intercept, ", 95 % CI = [",intercept- 2.*np.sqrt(var_it),",",intercept+2.*np.sqrt(var_it),"]")
    print( "R-squared", r_value**2)

# show models and best fit
    tt=xx
    tt.sort()
    fitx=np.arange(float(tt[0])-0.1,float(tt[-1])+0.1,0.1,dtype=float)
    fity=intercept + fitx*gradient
    #fityy=intercept + fitx
    fityyy= intercept+ 7./6.*fitx

    plt.plot(fitx,fity,'k-', linewidth=2, alpha=0.6,label=r'$\beta=??$, $r^2=??$, p-value $<1.e^{-20}$')
#plt.plot(fitx,fityy,'k-', linewidth=2, alpha=0.5,label=r'$\beta=??$, $r^2=??$, p-value $<1.e^{-20}$')
#plt.plot(fitx,fityyy,'y-', linewidth=6, alpha=0.5,label=r'$\beta=??$, $r^2=??$, p-value $<1.e^{-20}$')

plt.plot(avx,avy,'s',color='yellow',markeredgecolor='black',markeredgewidth=0.2,alpha=0.8)

plt.ylabel(r'$\log G_i$',fontsize=20)
plt.xlabel(r'$\log N_i$',fontsize=20)
plt.tight_layout()
#plt.show()
plt.savefig('GDP_scaling_over_time.pdf', format='pdf')




