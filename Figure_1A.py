import csv
import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib.colors as colors
from colorsys import hsv_to_rgb
import datetime as dt


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


yall=[]
iall=[]
gall=[]
poptot=[]
avx=[]
avy=[]

xx_tot=[]
yy_tot=[]


f=open('WB_data_clean.csv', 'r',encoding='latin-1')
reader=csv.reader(f,delimiter=',')

count=0
c2=0
c3=0
c4=0
nation=[]
GDP=[]
pop=[]
urban=[]
urbanr=[]
urban1m=[]
imports=[]
exports=[]


# possible countries:
countries = ['Afghanistan', 'Albania', 'Algeria','Angola', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bangladesh',  'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon', 'Canada', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Congo, Dem. Rep.', 'Congo, Rep.', 'Costa Rica', "Cote d'Ivoire", 'Croatia', 'Cuba', 'Cyprus', 'Czech Republic', 'Denmark', 'Ecuador', 'Egypt, Arab Rep.', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Finland','France', 'Gabon', 'Gambia, The', 'Georgia', 'Germany', 'Ghana','Greece','Grenada','Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran, Islamic Rep.', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Korea, Rep.', 'Kyrgyz Republic', 'Lao PDR', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Lithuania', 'Luxembourg', 'Macedonia, FYR', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Mauritania', 'Mauritius', 'Mexico', 'Moldova', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Norway', 'Pakistan', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto Rico', 'Romania', 'Russian Federation', 'Rwanda', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Slovak Republic', 'Slovenia', 'South Africa', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 'Switzerland', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Tunisia', 'Turkey', 'Turkmenistan', 'Uganda', 'Ukraine', 'United Kingdom', 'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela, RB', 'Vietnam', 'Yemen, Rep.', 'Zambia', 'Zimbabwe']


#countries=['United Kingdom', 'United States','Brazil','China','India','Argentina','Indonesia']


norm = colors.Normalize(vmin=1, vmax=2*len(countries))
sm = cm.ScalarMappable(norm, cmap=cm.Paired)
cnt = 1

names=[]
gradients=[]
intercepts=[]
xtot=[]
ytot=[]
ymin=1960
ymax=2017
ymin=ymin-1960
ymax=ymax-1960

for country in countries:
    print(country)
    f=open('WB_data_clean.csv', 'r',encoding='latin-1')
    reader=csv.reader(f,delimiter=',')
    
    count=0
    c2=0
    c3=0
    c4=0
    
    nation=[]
    GDP=[]  # GDP in constant 2010 $
    pop=[]
    urban=[] # urban population
    urbanr=[] # urban population (fraction)
    imports=[]
    exports=[]
    
    for row in reader:
    
        if (country==row[0] and row[2]=='Urban population'):
            for i in range(4,len(row)):
                urban.append( (row[i].replace(",","")) )
                count+=1

        if (country==row[0] and row[2]=='Urban population (% of total)'):   
    # variables = city, nation, population, GDP, Urbanized area,employment,patents,
            for i in range(4,len(row)):
                urbanr.append( (row[i].replace(",","")) )

        if (country==row[0] and row[2]=='GDP (constant 2010 US$)'):
            for i in range(4,len(row)):
                GDP.append((row[i].replace(",","")))
            c3+=1

    x=[]
    y=[]

    for i in range(len(urban)):
        if (urban[i] !='..' and GDP[i] !='..'):  # if not empty.
            #print(urban[i],GDP[i])
            aux=float(urbanr[i]) # fraction of population in urban
            aux2 = float(urban[i])/float(urbanr[i])*100. # this is the total population.
            aux3=float(urban[i])  # total urban population
            #aux4=float(urban1m[i])
            pop.append(aux)
# what is plotted:
            x.append(aux)
            y.append( float(GDP[i])/float(aux2))  # GDP per capita
    xlog=x #np.log10(x)
    ylog=np.log10(y)
    xx=xlog #[ymin:ymax]
    yy=ylog #-ylog[0] #[ymin:ymax] #-ylog[0]  $ need to be careful with what years this is available for
    for i in range(len(xx)):
        xtot.append(xx[i])
        ytot.append(yy[i])

    print("There are,",count,"years of data")
    if (len(xx)<70):
        edge_color, color = sm.to_rgba(cnt), sm.to_rgba(cnt+1)
        edge_color=color
        cnt += 2
        plt.plot(xx,yy,marker='o',ms=3,ls='None',c=color,markeredgecolor=edge_color,markeredgewidth=1,alpha=0.6,label=country)

        #print('Totals', len(xx), len(yy))
        gradient, intercept, r_value, var_gr, var_it = linreg(xx,yy)

        #print(country)
        #print("Gradient=", gradient, ", 95 % CI = [",gradient- 2.*np.sqrt(var_gr),",",gradient+2.*np.sqrt(var_gr),"]")
        #print("intercept=", intercept, ", 95 % CI = [",intercept- 2.*np.sqrt(var_it),",",intercept+2.*np.sqrt(var_it),"]")
        #print("R-squared", r_value**2)
        names.append(country)
        gradients.append(gradient)
        intercepts.append(intercept)
    
# show models and best fit
        tt=xx
        tt.sort()
        fitx=np.arange(float(tt[0])-0.25,float(tt[-1])+0.5,0.1,dtype=float)
        fity=intercept + gradient*fitx

        #plt.plot(fitx,fity,'r-', linewidth=2, alpha=0.5)
        f.close()
gradient, intercept, r_value, var_gr, var_it = linreg(xtot,ytot)
print("Global Fit")
print("Gradient=", gradient, ", 95 % CI = [",gradient- 2.*np.sqrt(var_gr),",",gradient+2.*np.sqrt(var_gr),"]")
print("intercept=", intercept, ", 95 % CI = [",intercept- 2.*np.sqrt(var_it),",",intercept+2.*np.sqrt(var_it),"]")
print("R-squared", r_value**2)
tt=xtot
tt.sort()
fitx=np.arange(float(tt[0])-0.25,float(tt[-1])+0.5,0.1,dtype=float)
fity=intercept + gradient*fitx
plt.plot(fitx,fity,'k-', linewidth=5, alpha=0.6)

xdim = np.sum(plt.xlim())/2
ydim = np.min(plt.ylim())*1.01
#plt.text(xdim, ydim, dt.datetime.now(), ha='right', va='baseline')
plt.ylim(2,5.5)

plt.ylabel('$\log_{10} \ g \ $(2010)',fontsize=20)
plt.xlabel('Percent Urban',fontsize=20)
#plt.legend()
#plt.show()
plt.savefig('WB_Trajectories_Global_Fit.pdf', format='pdf')






