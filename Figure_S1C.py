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

nn=16
popav=np.zeros(383)

for ii in range(2,nn): # each year

    f=open('real_GDP.csv', 'r')
    reader=csv.reader(f,delimiter=',')
    GDP=[] # each year, across cities
    fips=[]

    for row in reader:
        if (row[0].isdigit() and int(row[0])>10000):
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
    for i in range(len(GDP)): # over cities
        if (fips[i]==fipspc[i]):
            pop.append(GDP[i]/GDPpc[i]*10**6)
            popav[i]+=(GDP[i]/GDPpc[i]*10**6)
        else:
            print (fips[i])

popav=popav/float(nn)



fff=open('percapita_real_GDP_change_MSAs.csv', 'r')
readerfff=csv.reader(fff,delimiter=',')

gav=[]
gstd=[]
for row in readerfff:
    if (row[1]!=[] and row[0].isdigit() and int(row[0])>10000):
        pchange=[]
        for ii in range(2,18):
            pchange.append(float(row[ii]))
    
        gamma_av=np.mean(pchange)
        gamma_std=np.std(pchange)
        if (gamma_av>4.5 or gamma_std > 10.):
            print(row[1],gamma_av,gamma_std)
        gav.append(gamma_av)
        gstd.append(gamma_std)



#### tests: fit and quartiles

xx=np.log10(popav)
yy=gav # gav

gradient, intercept, r_value, var_gr, var_it = linreg(xx,yy)
print( "Gradient=", gradient, ", 95 % CI = [",gradient- 2.*np.sqrt(var_gr),",",gradient+2.*np.sqrt(var_gr),"]")
print( "intercept=", intercept, ", 95 % CI = [",intercept- 2.*np.sqrt(var_it),",",intercept+2.*np.sqrt(var_it),"]")
print( "R-squared", r_value**2)


plt.plot(xx,yy,'bo',alpha=0.5)
xs=xx
ys=gav  # gav


tt=xx
tt.sort()
fitx=np.arange(float(tt[0])-0.1,float(tt[-1])+0.1,0.1,dtype=float)
fity=intercept + fitx*gradient
#fityy=intercept + fitx
fityyy= intercept+ 7./6.*fitx
    
#plt.plot(fitx,fity,'k-', linewidth=2, alpha=0.6,label=r'$\beta=??$, $r^2=??$, p-value $<1.e^{-20}$')

md=np.median(xs)
qq=[]
for i in range(len(xs)):
    if (xs[i]<= md):
        qq.append(xs[i])

mqq=np.median(qq)

qq=[]
for i in range(len(xs)):
    if (xs[i]> md):
        qq.append(xs[i])

Mqq=np.median(qq)
#print('qq',mqq,md,Mqq)

sigma_m=0.
sigma_mdm=0.
sigma_mdM=0.
sigma_M=0.

x_m=0.
x_mdm=0.
x_mdM=0.
x_M=0.

y_m=0.
y_mdm=0.
y_mdM=0.
y_M=0.


n_m=0.
n_mdm=0.
n_mdM=0.
n_M=0.

for i in range(len(xs)):
    if (xs[i]<=mqq):
        y_m+=ys[i]
        sigma_m+=ys[i]**2
        x_m+=xs[i]
        n_m+=1.
    if (xs[i]>mqq and xs[i]<=md):
        y_mdm+=ys[i]
        sigma_mdm+=ys[i]**2
        x_mdm+=xs[i]
        n_mdm+=1.
    
    if (xs[i]>md and xs[i]<=Mqq):
        y_mdM+=ys[i]
        sigma_mdM+=ys[i]**2
        x_mdM+=xs[i]
        n_mdM+=1.
    
    if (xs[i]>Mqq):
        y_M+=ys[i]
        sigma_M+=ys[i]**2
        x_M+=xs[i]
        n_M+=1.

sigma_m=np.sqrt(sigma_m/n_m)
sigma_mdm=np.sqrt(sigma_mdm/n_mdm)
sigma_mdM=np.sqrt(sigma_mdM/n_mdM)
sigma_M=np.sqrt(sigma_M/n_M)
x_m=x_m/n_m
x_mdm=x_mdm/n_mdm
x_mdM=x_mdM/n_mdM
x_M=x_M/n_M

y_m=y_m/n_m
y_mdm=y_mdm/n_mdm
y_mdM=y_mdM/n_mdM
y_M=y_M/n_M


print('sigmas',sigma_m,sigma_mdm,sigma_mdM,sigma_M)
print('x',x_m,x_mdm,x_mdM,x_M)

plt.plot((min(xx)-0.2,max(xx)+0.2),(0.105,0.105),'k--')
plt.plot((x_m,x_mdm,x_mdM,x_M),(y_m,y_mdm,y_mdM,y_M),'ro',ms=10)
plt.plot((mqq,mqq),(-2.0,5),'r--')
plt.plot((md,md),(-2.0,5),'r--')
plt.plot((Mqq,Mqq),(-2.0,5),'r--')


plt.ylabel('${\gamma_i}$',fontsize=20)
plt.xlabel('$\log_{10} \ N_i$',fontsize=20)
plt.tight_layout()
#plt.show()
plt.savefig('GDPpc_growth_rates_average.pdf', format='pdf')


