import csv
import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib.colors as colors
from colorsys import hsv_to_rgb
import datetime as dt

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


f=open('urbanization-vs-gdp.csv', 'r',encoding='latin-1')
reader=csv.reader(f,delimiter=',')

window=30

norm = colors.Normalize(vmin=1, vmax=2*165)
sm = cm.ScalarMappable(norm, cmap=cm.Paired)
cnt = 0
c=0
nation=''
uall=[]
gall=[]
xall=[]
yall=[]

nav=20
avxx=np.zeros(nav)
avyy=np.zeros(nav)
avcount=np.zeros(nav)

for row in reader:
    if (row[1]!='' and row[3]!='' and row[4]!='' and row[4].isdigit() and float(row[3])<99. ):
        #print(row[2],row[3],row[4])
        if (nation!=row[0]):
            #if (cnt>0 and ur[-1]<95 and nation=='France'): # and len(ur)>60):
            #or nation=='South Korea' or nation=='United States' or nation=='France' or nation=='Portugal' or nation=='China' or nation=='Germany' or nation=='Japan' or nation=='Brazil'
            if (cnt>0):# and nation=='United States' or  nation=='South Korea'): # and len(ur)>60):
                #print("There are,",cnt,"years of data")
                xx=ur
                yy=np.log10(gdp)
                edge_color, color = sm.to_rgba(c), sm.to_rgba(c+1)
                edge_color=color
                cnt += 1
                c += 2
                
                for i in range(len(ur)):
                    if (yy[i]<2.5):
                        print(str(nation),year[i])
                    xall.append(xx[i])
                    yall.append(yy[i])
                
                #plt.plot(xx,yy,marker='o',ms=3,ls='-',lw=2,c=color,markeredgecolor=edge_color,markeredgewidth=1,alpha=0.99,label=str(nation))
                #plt.plot(year,yy,marker='o',ms=3,ls='-',lw=2,c=color,markeredgecolor=edge_color,markeredgewidth=1,alpha=0.99,label=str(nation))
                #plt.plot(xx,yy,marker='None',ms=3,ls='-',c=color,markeredgecolor=edge_color,markeredgewidth=1,alpha=0.6,label=str(nation))
                plt.plot(xx,yy,marker='o',ms=3,ls='None',c=color,markeredgecolor=edge_color,markeredgewidth=1,alpha=0.6,label=str(nation))
                #plt.plot(dxx,dyy,marker='o',ms=3,ls='-',c=color,markeredgecolor=edge_color,markeredgewidth=1,alpha=0.6)
                #plt.plot(xx[len(yy)-len(rm):],rm,marker='None',ms=3,ls='-',c=color,markeredgecolor=edge_color,markeredgewidth=1,alpha=0.6,label=str(nation))
            gdp=[]
            ur=[]
            year=[]
            pop=[]
            gdp.append(float(row[4]))
            ur.append(float(row[3]))
            year.append(int(row[2]))
            
            nation=row[0]
            #print(row[0],row[1],row[2],row[3],row[4])
            cnt=0
        else:
            ind=int(float(row[3])/100.*float(nav))
            #print(float(row[3])/float(nav),ind)
            avxx[ind]+=float(row[3])
            avyy[ind]+=float(row[4])
            avcount[ind]+=1
            
            gdp.append(float(row[4]))
            ur.append(float(row[3]))
            year.append(int(row[2]))
           
            cnt+=1

for i in range(len(avcount)):
    avxx[i]=avxx[i]/float(avcount[i])
    avyy[i]=avyy[i]/float(avcount[i])

avyy=np.log10(avyy)

print("There are,",cnt,"years of data")
print ('There are ',c,'nations')

gradient, intercept, r_value, var_gr, var_it = linreg(xall,yall)
print("Gradient=", gradient, ", 95 % CI = [",gradient- 2.*np.sqrt(var_gr),",",gradient+2.*np.sqrt(var_gr),"]")
print("intercept=", intercept, ", 95 % CI = [",intercept- 2.*np.sqrt(var_it),",",intercept+2.*np.sqrt(var_it),"]")
print("R-squared", r_value**2)
    
# show models and best fit
tt=xall
tt.sort()
fitx=np.arange(float(tt[0])+0.01,float(tt[-1])+0.01,0.1,dtype=float)

fity=intercept + gradient*fitx



#fityy = intercept +0.12  ( (fitx/100.)*0.012005158839)*np.log( fitx/(98.95-fitx) )/np.log(10.)/0.1 + 0.0137532247855  )*fitx

#b0=2.596625480/np.log(10.) # = 1.277 --> /100 --> 0.01277 ~0.013
b0=3.2/np.log(10.)
b1=0.012
b2=0.
#b1=0.0140394490192-0.00355198773142
#b2=0.00355198773142
uM=100.


kk=1/30.
fityy = intercept +b0*fitx/100. + 0.12 +( (fitx/100.)*b1 +b2 )*np.log(fitx/(uM-fitx))/np.log(10.)/kk

kk=0.05
fityyy = intercept +b0*fitx/100. + 0.12 +( (fitx/100.)*b1 +b2 )*np.log(fitx/(uM-fitx))/np.log(10.)/kk

kk=0.1
fityyyy = intercept +b0*fitx/100. + 0.12 +( (fitx/100.)*b1 +b2 )*np.log(fitx/(uM-fitx))/np.log(10.)/kk


plt.plot(fitx,fity,'k-', linewidth=5, alpha=0.5)

plt.plot(fitx,fityy,'b-', linewidth=2, alpha=1.0)
plt.plot(fitx,fityyy,'b-', linewidth=2, alpha=1.0)
plt.plot(fitx,fityyyy,'b-', linewidth=2, alpha=1.0)


#plt.plot(fitx,fityyy,'r-', linewidth=2, alpha=1.0)


#plt.plot(avxx,avyy,'ro', ms=8, alpha=1.0)

plt.ylabel('$\log_{10} \ g$ (2011)',fontsize=20)
plt.xlabel('Percent Urban',fontsize=20)
plt.tight_layout()
#plt.legend()
#plt.show()
plt.savefig('Trajectories_Fit_OWID.pdf', format='pdf')






