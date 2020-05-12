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
from scipy.interpolate import interp1d
import matplotlib.ticker as ticker
from statsmodels.api import add_constant
from statsmodels.api import OLS

def reg_m(y, X):
    X = sm.add_constant(X)
    results = sm.OLS(y, X).fit
    return results

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

def fit(year,ur, gammaR, gammaU):
    return np.multiply(year,(gammaR +ur*(gammaU-gammaR)))
#    return 100.*np.arctan(k*0.001*(x-x0))

f=open('../urbanization-vs-gdp.csv', 'r',encoding='latin-1')
reader=csv.reader(f,delimiter=',')

window=30

norm = colors.Normalize(vmin=1, vmax=2*5.)
sm = cm.ScalarMappable(norm, cmap=cm.Paired)
cnt = 0
c=0
nation=''
uall=[]
gall=[]
xall=[]
yall=[]

nb=8
bins=np.zeros(nb)
ubins=np.zeros(nb)
nbins=np.zeros(nb)
ybins=np.zeros(nb)

fig, ax = plt.subplots()
for row in reader:
    if (row[1]!='' and row[3]!='' and row[4]!='' and row[4].isdigit() and row[5]!=''):
        if (nation!=row[0]):
            if (cnt>0): #and nation=='Spain' or nation=='France' or nation=='United States' or nation=='Japan' or nation=='South Africa' or nation=='United Kingdom'): # and len(ur)>60):
                xx=gdp
                yy=np.log(gdp) # this is the natural log !! (different from log 10)
                duu=[]
                dyy=[]
                dxx=[]
                duu=[]
                dur=[]
                for i in range(len(ur)-1):
                    dt=year[i+1]-year[i]
                    aux1 =(yy[i+1]-yy[i]) /dt # this is the growth rate (difference of log G in natural! log)
                    aux2=(ur[i+1]-ur[i]) /dt # this is in percentage units
                    aux3=year[i]-year[0]
                    #print(year[i])
                    dyy.append(aux1)
                    dxx.append( (year[i+1]+year[i])/2. )
                    duu.append((ur[i+1]+ur[i])/2.)
                    dur.append(aux2)
                    
                    aa=int(round((ur[i+1]+ur[i])/2.)*nb/100.) # this is just the bin index, needs to convert percentage urbanization to fraction, hence /100
                    if (aa !=nb):
                        ubins[aa]=ubins[aa]+aux2 # the velocity of urbanization in percentage points per year
                        bins[aa]=bins[aa]+aux1 # the growth rate per year
                        ybins[aa]=ybins[aa]+aux3 # this is the time difference in year
                        nbins[aa]+=1
                    
                    uall.append(xx[i])
                    gall.append(yy[i])
                edge_color, color = sm.to_rgba(cnt), sm.to_rgba(c+1)
                edge_color=color
                c += 1
                aux=year[0]
                aux1=xx[0]
                for i in range(len(ur)):
                    xall.append(xx[i])
                    yall.append(yy[i])
                    year[i]=year[i]-aux
                
                f = interp1d(duu, dyy)
                xnew = np.linspace(duu[0], duu[-1], num=30, endpoint=True)
                plt.plot((0.,100.),(0.,0.),'k-')
                
        
            gdp=[]
            ur=[]
            year=[]
            pop=[]
            gdp.append(float(row[4]))
            ur.append(float(row[3]))
            year.append(int(row[2]))
            pop.append(float(row[5]))
            nation=row[0]
            #print(row[0],row[1],row[2],row[3],row[4])
            cnt=0
        else:
            gdp.append(float(row[4]))
            ur.append(float(row[3]))
            year.append(int(row[2]))
            pop.append(float(row[5]))
            cnt+=1
print("There are,",cnt,"years of data")

xbins=[]
for i in range(nb):
    bins[i]=bins[i]/float(nbins[i])
    ubins[i]=ubins[i]/float(100.*nbins[i]) # this makes the velocity be in fraction, not percentage
    ybins[i]=ybins[i]/float(nbins[i])
    xbins.append((i+0.5)*100./float(nb))
ybins=ybins-ybins[0]

# This fits the GDPpc growth rate agains urbanization rate and it time derivative.
x1=np.true_divide(xbins,100.) # to normalize u to 1, rather than 100.
x2=[]
for i in range(len(x1)):
    x2.append(x1[i]+ybins[i]*ubins[i])  # is this right? it is in fraction units not percentage.

print('x2=',x2)
print('x1=',x1)
print('ubins=',ubins)
print('ybins=',ybins)
#xx=[x1, ubins] # to input to fit: x1 = urbanization rate; ubins = velocity of urbanization rate
xx=[ubins,x2]
X = np.array(xx).T # to format the vector correctly

model = OLS(bins,X)
results = model.fit()
print(results.summary())
#print(results.params[2]/results.params[1])

fit=[]
fitx=[]
for i in range(nb):
    fit.append(results.params[1]*x2[i]+results.params[0]*ubins[i] )
    fitx.append(results.params[1]*x1[i])

plt.plot(xbins,bins,'ro-',label='growth rate')
plt.plot(xbins,fit,'go-',label='best fit')

plt.plot(xbins,results.params[0]*ubins,'bo-',label=r'$b_0 v_u + b_2$')
plt.plot(xbins,fitx,color='cyan',label=r'$b_1 u$')

#ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x +aux))
#ax.xaxis.set_major_formatter(ticks_x)

#print('gamma_R=',results.params[0])
print('gamma_U=',results.params[1])
print('ln  g_U/g_R =',results.params[0]) # this is  in  natural log units, not log 10...
print('b_1=',results.params[0]/100./np.log(10.))


plt.ylabel(r'$\frac{\Delta \ln g}{\Delta t}$ (2010)',fontsize=20)
plt.xlabel(r'urbanization rate, $u$',fontsize=20)
plt.tight_layout()
plt.legend()
#plt.show()
plt.savefig('GDPpc_Av_Growth_Rate_Fit_no_constant.pdf', format='pdf')

