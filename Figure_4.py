import csv
import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib.colors as colors
from colorsys import hsv_to_rgb
import datetime as dt
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy import interpolate
import matplotlib.ticker as ticker

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def lin_fit(x, y):
    '''Fits a linear fit of the form mx+b to the data'''
    fitfunc = lambda params, x: params[0] * x    #create fitting function of form mx+b
    errfunc = lambda p, x, y: fitfunc(p, x) - y              #create error function for least squares fit

    init_a = 0.5                            #find initial value for a (gradient)
    init_p = np.array((init_a))  #bundle initial values in initial parameters

    #calculate best fitting parameters (i.e. m and b) using the error function
    p1, success = scipy.optimize.leastsq(errfunc, init_p.copy(), args = (x, y))
    f = fitfunc(p1, x)          #create a fit with those parameters
    return p1, f

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
    return a, b, RR, Var_a, Var_b

def sigmoid(x, aa, k, x0):
    return 80./(1 + np.exp(-k*0.01*(x-x0))) # final amplitude (here 80) needs to be adjusted for different nations

f=open('urbanization-vs-gdp.csv', 'r',encoding='latin-1')
reader=csv.reader(f,delimiter=',')

window=20

norm = colors.Normalize(vmin=1, vmax=2*5.)
sm = cm.ScalarMappable(norm, cmap=cm.Paired)
cnt = 0
c=0
nation=''
uall=[]
gall=[]
xall=[]
yall=[]
fig, ax = plt.subplots()
for row in reader:
    if (row[1]!='' and row[3]!='' and row[4]!='' and row[4].isdigit() and row[5]!=''):
        if (nation!=row[0]):
            #or nation=='South Korea' or nation=='United States' or nation=='France' or nation=='Portugal' or nation=='China' or nation=='Germany' or nation=='Japan' or nation=='Brazil'
            if (cnt>0 and nation=='United States'): # and len(ur)>60):
                #print("There are,",cnt,"years of data")
                xx=ur
                yy=np.log10(gdp)
                dyy=[]
                dxx=[]
                for i in range(len(ur)-1):
                    dt=year[i+1]-year[i]
                    aux =(yy[i+1]-yy[i])/dt
                    dyy.append(aux)
                    dxx.append( (ur[i+1]+ur[i])/2. )
                    uall.append(xx[i])
                    gall.append(yy[i])
                #print(np.mean(dyy))
                edge_color, color = sm.to_rgba(cnt), sm.to_rgba(c+1)
                edge_color=color
                c += 1
                rm=running_mean(yy, window)
                aux=year[0]
                aux1=xx[0]
                for i in range(len(ur)):
                    xall.append(xx[i])
                    yall.append(yy[i])
                    year[i]=year[i]-aux
                    xx[i]=xx[i]-aux1
                
                
                tck = interpolate.splrep(year, xx, s=0)
                xnew = np.arange(0, year[-1], year[-1]/10)
                ynew = interpolate.splev(xnew, tck, der=0)
                plt.plot(xnew, ynew,'ro')
                
                #popt, pcov = curve_fit(sigmoid, xnew, ynew)
                
                popt, pcov = curve_fit(sigmoid, year, xx)
                
                print('1/k=',100./popt[1]) # this is the 1/k time scale
                print(popt)
                
                x = np.linspace(year[0], year[-1], 100)
                y = sigmoid(x, *popt)
                plt.plot(x,y, label='fit')
                plt.plot(year,xx,marker='o',ms=3,ls='-',lw=2,c=color,markeredgecolor=edge_color,markeredgewidth=1,alpha=0.6,label=str(nation))
            gdp=[]
            ur=[]
            year=[]
            pop=[]
            gdp.append(float(row[4]))
            ur.append(float(row[3]))
            year.append(int(row[2]))
            pop.append(float(row[5]))
            nation=row[0]
            cnt=0
        else:
            gdp.append(float(row[4]))
            ur.append(float(row[3]))
            year.append(int(row[2]))
            pop.append(float(row[5]))
            cnt+=1
print("There are,",cnt,"years of data")

ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x +aux))
ax.xaxis.set_major_formatter(ticks_x)

ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y +aux1))
ax.yaxis.set_major_formatter(ticks_y)

plt.ylabel('Urbanization Rate',fontsize=20)
plt.xlabel('Year',fontsize=20)
plt.tight_layout()
plt.legend()
plt.savefig('Urbanization_Trajectory_Fit_United_States.pdf', format='pdf')






