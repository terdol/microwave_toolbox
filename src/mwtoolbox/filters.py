#-*-coding:utf-8-*-
from .myconstants import *
from .components import z_wg_te10
from scipy import sin,sinh,arcsin,arcsinh,cos,cosh,arccos,arccosh,tan,arctan,tanh,arctanh,log,log10,power
from scipy.special import *
from scipy.optimize import brentq
from scipy.integrate import quad
from string import *
import re
import numpy as np
from numpy.lib.scimath import sqrt as csqrt
from .network import *
co=speed_of_light_in_freespace.simplified.magnitude
eta0=free_space_wave_impedance.simplified.magnitude
mu0=free_space_permeability.simplified.magnitude
eps0=free_space_permittivity.simplified.magnitude

def Z_WG_TE10(er,a,b,freq):
    return z_wg_te10(er,a,b,freq,1)

def EWG_ABCD(a,b,er,length,frek):
    #Referans:"The Design of Evanescent Mode Waveguide Bandpass Filters for a Prescribed Insertion Loss Characteristic.pdf"
    # Model= Xp1,Xs1,Xp1 ya da Xs2,Xp2,Xs2 (p: shunt, s: series)
    #Zo=jXo
    lcutoff=2*a
    wavelength=co/csqrt(er)/frek
    Xo=abs(Z_WG_TE10(er,a,b,frek))
    gamma=2.*pi/wavelength*csqrt(((wavelength/lcutoff))**2.0-1.)
    Xs1=Xo*sinh(gamma*length)
    Xs2=Xo*tanh(gamma*length/2.0)
    Xp1=(Xo/tanh(gamma*length/2.0))
    Xp2=(Xo/sinh(gamma*length))
    networks=[]
    #networks.append(shunt_z(-1.0j*Xo*tanh(gamma*length)))
    networks.append(shunt_z(1.0j*Xp1))
    networks.append(series_z(1.0j*Xs1))
    networks.append(shunt_z(1.0j*Xp1))
    #networks.append(shunt_z(-1.0j*Xo*tanh(gamma*length)))
    return cascade_networks(networks)

def MinimumButterworthFilterDegree(L,fstop):
    #KAYNAK:Microstrip Filters for RF Microwave Applications s.42, fstop: normalize frekans, L: fstoptaki bastirma (dB)
    return int((log10(pow(10.0,0.1*L)-1.0)/(2.0*log10(fstop))))+1

def ButterworthFilterPrototype(n):
    #KAYNAK:Microstrip Filters for RF Microwave Applications s.41, fstop'taki bastirmayi da soyler, fstop: normalize frekans
    g=list(range(n+2))
    g[0]=g[n+1]=1.0
    g[1:n+1]=2.*sin((2*np.array(list(range(1, n+1)))-1)*pi/2./n)
    return g, -10.0*log10(1.0+pow(fstop,2*n))

def MinimumChebyshevFilterDegree(Lar,Las,fstop):
    #KAYNAK:Microstrip Filters for RF Microwave Applications s.41, fstop: normalize frekans, Las: fstoptaki bastirma (dB), Lar: bant ici ripple (dB)
    temp=csqrt(((pow(10.0,0.1*Las)-1)/(pow(10.0,0.1*Lar)-1)))
    return int((arccosh(temp)/arccosh(fstop)))+1

def ChebyshevFilterPrototype(n,Lar):
    #n-odd, KAYNAK:Microstrip Filters for RF Microwave Applications s.42, Lar: passband ripple in dB
    beta=-log(tanh((Lar/17.37)))
    gamma=sinh(beta/2/n)
    g=list(range(n+2))
    g[0]=g[n+1]=1.0
    g[1]=2/gamma*sin(0.5*pi/n)
    i=2
    while i<n+1:
        g[i]=1./g[i-1]*4*sin((2*i-1)*pi/2/n)*sin((2*i-3)*pi/2/n)/(gamma**2+sin((i-1)*pi/n)**2)
        i=i+1
    return g
def LPFilterFromPrototype(g,Zo,fc,type=1):
    #n-odd, KAYNAK:Microstrip Filters for RF Microwave Applications s.42
    #type=1, starts with L
    #type=2, starts with C
    n=len(g)-2
    g=g[:1]+[k/2/pi/fc for k in g[1:-1]]+g[-1:]
    sonuc=""
    if type==1:
        for i in range(1,n+1):
            if i%2==1:
                g[i]=g[i]*Zo
                sonuc=sonuc+str(i)+"-Series L- "+str(g[i])+" H\n"
            else:
                g[i]=(g[i]/Zo)
                sonuc=sonuc+str(i)+"-Shunt C-"+str(g[i])+" F\n"
    elif type==2:
        for i in range(1,n+1):
            if i%2==1:
                g[i]=(g[i]/Zo)
                sonuc=sonuc+str(i)+"-Shunt C-"+str(g[i])+" F\n"
            else:
                g[i]=g[i]*Zo
                sonuc=sonuc+str(i)+"-Series L-"+str(g[i])+" H\n"
    return sonuc
def HPFilterFromPrototype(g,Zo,fc,type=1):
    #n-odd, KAYNAK:Microstrip Filters for RF Microwave Applications s.42
    #type=1, starts with L
    #type=2, starts with C
    n=len(g)-2
    g=g[:1]+[1./2./pi/fc/k for k in g[1:-1]]+g[-1:]
    sonuc=""
    if type==1:
        for i in range(1,n+1):
            if i%2==1:
                g[i]=g[i]*Zo
                sonuc=sonuc+str(i)+"-Shunt L-"+str(g[i])+" H\n"
            else:
                g[i]=(g[i]/Zo)
                sonuc=sonuc+str(i)+"-Series C-"+str(g[i])+" F\n"
    elif type==2:
        for i in range(1,n+1):
            if i%2==1:
                g[i]=(g[i]/Zo)
                sonuc=sonuc+str(i)+"-Series C-"+str(g[i])+" F\n"
            else:
                g[i]=g[i]*Zo
                sonuc= sonuc+str(i)+"-Shunt L-"+str(g[i])+" H\n"
    return sonuc
def BPFilterFromPrototype(g,Zo,fc,FBW,type=1):
    #n-odd, KAYNAK:Microstrip Filters for RF Microwave Applications s.42
    #type=1, starts with L
    #type=2, starts with C
    n=len(g)-2
    sonuc=""
    if type==1:
        for i in range(1,n+1):
            if i%2==1:
                sonuc=sonuc+" Ls"+str(i)+" "+str(g[i]*Zo/FBW/2./pi/fc)+"H\t Cs"+str(i)+" "+str(FBW/2./pi/fc/Zo/g[i])+"F\n"
            else:
                sonuc=sonuc+" Lp"+str(i)+" "+str(Zo*FBW/2./pi/fc/g[i])+"H\t Cp"+str(i)+" "+str(1./FBW/2./pi/fc/Zo*g[i])+"F\n"
    elif type==2:
        for i in range(1,n+1):
            if i%2==1:
                sonuc=sonuc+" Lp"+str(i)+" "+str(Zo*FBW/2./pi/fc/g[i])+"H\t Cp"+str(i)+" "+str(1./FBW/2./pi/fc/Zo*g[i])+"F\n"
            else:
                sonuc=sonuc+" Ls"+str(i)+" "+str(g[i]*Zo/FBW/2./pi/fc)+"H\t Cs"+str(i)+" "+str(FBW/2./pi/fc/Zo/g[i])+"F\n"
    return sonuc

def BSFilterFromPrototype(g,Zo,fc,FBW,type=1):
    #n-odd, KAYNAK:Microstrip Filters for RF Microwave Applications s.42
    #type=1, starts with L
    #type=2, starts with C
    n=len(g)-2
    sonuc=""
    if type==1:
        for i in range(1,n+1):
            if i%2==1:
                sonuc=sonuc+" Lp"+str(i)+" "+str(g[i]*Zo*FBW/2./pi/fc)+"H\t Cp"+str(i)+" "+str(1./FBW/2./pi/fc/Zo/g[i])+"F\n"
            else:
                sonuc=sonuc+" Ls"+str(i)+" "+str(Zo/FBW/2./pi/fc/g[i])+"H\t Cs"+str(i)+" "+str(FBW/2./pi/fc/Zo*g[i])+"F\n"
    elif type==2:
        for i in range(1,n+1):
            if i%2==1:
                sonuc=sonuc+" Ls"+str(i)+" "+str(Zo/FBW/2./pi/fc/g[i])+"H\t Cs"+str(i)+" "+str(FBW/2./pi/fc/Zo*g[i])+"F\n"
            else:
                sonuc=sonuc+" Lp"+str(i)+" "+str(g[i]*Zo*FBW/2./pi/fc)+"H\t Cp"+str(i)+" "+str(1./FBW/2./pi/fc/Zo/g[i])+"F\n"
    return sonuc

def ChebyshevSteppedImpedanceLPFilter(Zo,maxreturnloss,N,theta):
    #alpha=sin(theta), theta=ue'lerin bant kenarindaki elektriksel uzunlugu
    rip=csqrt((1./(10.**((maxreturnloss/10.0))-1)))
    alpha=sin(theta/180.*pi)
    RL=1
    Z=[]
    if N%2==0:
        print("N should be an odd number.")
        RL=((csqrt(1+rip**2)-rip)/(csqrt(1+rip**2)+rip))
    eta=sinh(1./N*arcsinh((1./rip)))
    A=(1./eta**2.)
    for i in range(1,N+1):
        x=(2*sin((2*i-1)*pi/2./N)/alpha-alpha/4.*(((eta**2+sin(i*pi/N)**2)/sin((2*i+1)*pi/2./N))+((eta**2+sin((i-1)*pi/N)**2)/sin((2*i-3)*pi/2./N))))
        if i%2==0:
            A=1./A/(eta**2+sin((i-1)*pi/N)**2)
            x=x*A*eta
            Z.append(Zo*x)
        else:
            A=1./A/(eta**2+sin((i-1)*pi/N)**2)
            x=x*A/eta
            Z.append((Zo/x))
    return Z

def InductivePostWGFilter(er, a, b, maxreturnloss,N, d,  x,  f1, f2):
    #kaynak: theory and design of microwave filters s.220
    fcenter=((f1+f2)/2.)
    fc=co/csqrt(er)/2./a
    lg1=co/csqrt(er)/f1/csqrt(1.-((fc/f1))**2.)
    lg2=co/csqrt(er)/f2/csqrt(1.-((fc/f2))**2.)
    lgo=((lg1+lg2)/2.)+1./pi*(lg1*cos(pi/2.*lg2/lg1)+lg2*cos(pi/2.*lg1/lg2))/(sin(pi/2.*lg2/lg1)+sin(pi/2.*lg1/lg2))
    alpha=(1./(lg1/lgo*sin(pi*lgo/lg1)))
    rip=csqrt((1./(10.**((maxreturnloss/10.0))-1)))
    eta=sinh(1./N*arcsinh((1./rip)))
    r=array(list(range(1, N+1)))
    Z=ones(N+2)
    K=ones(N+1)
    Z[1:N+1]=2*alpha/eta*sin((2*r-1)*pi/2./N) -0.25/eta/alpha*(((eta**2+sin(r*pi/N)**2)/sin((2*r+1)*pi/2./N))+((eta**2+sin((r-1)*pi/N)**2)/sin((2*r-3)*pi/2./N)))
    r=array(list(range( N+1)))
    K=(csqrt(eta**2+sin(r*pi/N)**2)/eta)
    for i in range(N+1):
        K[i]=(K[i]/csqrt(Z[i]*Z[i+1]))
    B=(1./K)-K
    L=1./2./pi/fcenter/B
    fi=ones(N)
    lengths=ones(N)
    for i in range(N):
        fi[i]=(pi-0.5*(arctan((2./B[i]))+arctan((2./B[i+1]))))/pi*180.
        lengths[i]=fi[i]/180./csqrt(4.*er*fcenter**2/co**2-(1./a**2))
    #constant diameter synthesis
    sonuc_x=[]
    def integrand(c):
        ind, cap, z=InductivePostInWaveguide(er, a, b, d, c, fcenter)
        return (ind/z)
    Lmin=integrand((a/2.0))
    for temp in L:
        if temp<Lmin:
            sonuc_x.append(0)
            print("Inductance truncated to the minimum achievable")
        else:
            sonuc_x.append(brentq(lambda x:integrand(x)-temp,(d/2.0),(a/2.),   maxiter=200))

    #constant offset synthesis
    sonuc_d=[]
    def integrand(c):
        ind, cap, z=InductivePostInWaveguide(er, a, b, c, x, fcenter)
        return (ind/z)
    for temp in L:
        try:
            sonuc_d.append(brentq(lambda x:integrand(x)-temp,(a/1000.),a,   maxiter=200))
        except:
            print("Error at calculating diameter, L="+str(temp))
    return (L,fi, sonuc_x, sonuc_d, lengths)

def filter_with_j_inverter(g,Zo,fcenter,fbw,caps):
    #order=N ise g'nin boyutu n+2 kapasite sayisi ise n'dir ve biz belirleriz.J-inverter sayisi n+1'dir
    inds=[]
    n=len(g)-2
    Yo=(1./Zo)
    jvalues=list(range(n+1))
    for i in range(1,n):
        jvalues[i]=fbw*2*pi*fcenter*csqrt(caps[i-1]*caps[i]/g[i]/g[i+1])
    jvalues[0]=csqrt(fbw*2*pi*fcenter*caps[0]*Yo/g[0]/g[1])
    jvalues[n]=csqrt(fbw*2*pi*fcenter*caps[n-1]*Yo/g[n]/g[n+1])
    inds=1./(2.*pi*fcenter)**2/array(caps)
    return jvalues,inds,caps

def filter_with_j_inverter2(g,Zo,fcenter,fbw,caps): #first and last inverters are eliminated
    #order=N ise g'nin boyutu n+2 kapasite sayisi ise n'dir ve biz belirleriz.J-inverter sayisi n+1'dir
    inds=[]
    n=len(g)-2
    Yo=(1./Zo)
    jvalues=list(range(n+1))
    for i in range(1,n):
        jvalues[i]=fbw*2*pi*fcenter*csqrt(caps[i-1]*caps[i]/g[i]/g[i+1])
    jvalues[0]=csqrt(fbw*2*pi*fcenter*caps[0]*Yo/g[0]/g[1])
    jvalues[n]=csqrt(fbw*2*pi*fcenter*caps[n-1]*Yo/g[n]/g[n+1])
    inds=1./(2.*pi*fcenter)**2/array(caps)
    return (jvalues[1:-1]/(Zo*Zo*jvalues[0]*jvalues[0])),inds*Zo*Zo*jvalues[0]*jvalues[0],(caps/(Zo*Zo*jvalues[0]*jvalues[0]))


def EvanescentWGFilter_3(g,n,Lj,a,a1,b,er, fcenter,fbw,alpha):
    #order=3,
    #n:WG junction transformer ratio (@fcenter)
    #Zj:abs(WG junction impedance)  (@fcenter)
    #alpha: J-coupled 3 order filtrede C2/C1
    #a: width of TE10 waveguide
    #a1: width of evanescent waveguide
    lcutoff=2*a1
    wavelength=co/csqrt(er)/fcenter
    w0=2.0*pi*fcenter
    Xo=abs(Z_WG_TE10(er,a1,b,fcenter))
    Zo=abs(Z_WG_TE10(er,a,b,fcenter))
    #Zo=100.0
    C2=g[0]*g[1]*alpha/fbw/w0/n/n/Zo
    C1=g[0]*g[1]/fbw/w0/n/n/Zo
    L2=1.0/w0/w0/C2
    L1=alpha*L2  #fbw*n*n*Zo/(w0*g0*g1)
    gamma=2.*pi/wavelength*csqrt(((wavelength/lcutoff))**2.0-1.+0.0j)
    J=g[0]*g[1]*csqrt(alpha/g[1]/g[2])/n/n/Zo
    length=(arcsinh(abs(1.0/Xo/J))/gamma)
    L=Xo*tanh(gamma*length)/w0
    L1=L1*L/(L-L1)
    L1=L1*Lj/(Lj-L1/n/n)
    L2=L2*Xo*tanh(gamma*length)/(Xo*tanh(gamma*length)-2*w0*L2)
    return (length,C1,C2,L1,L2,J)

def EvanescentWGFilter_4(g,n,Lj,a,a1,b,er, fcenter,fbw,alpha):
    #generalized version of EvanescentWGFilter_3
    #n:WG junction transformer ratio (@fcenter)
    #Zj:abs(WG junction impedance)  (@fcenter)
    #alpha: J-coupled 3 order filtrede C2/C1
    #a: width of TE10 waveguide
    #a1: width of evanescent waveguide
    lcutoff=2*a1
    wavelength=co/csqrt(er)/fcenter
    w0=2.0*pi*fcenter
    Xo=abs(Z_WG_TE10(er,a1,b,fcenter))
    Zo=abs(Z_WG_TE10(er,a,b,fcenter))
    #Zo=100.0
    C2=g[0]*g[1]*alpha/fbw/w0/n/n/Zo
    C1=g[0]*g[1]/fbw/w0/n/n/Zo
    L2=1.0/w0/w0/C2
    L1=alpha*L2  #fbw*n*n*Zo/(w0*g0*g1)
    gamma=2.*pi/wavelength*csqrt(((wavelength/lcutoff))**2.0-1.+0.0j)
    J=g[0]*g[1]*csqrt(alpha/g[1]/g[2])/n/n/Zo
    length=(arcsinh(abs(1.0/Xo/J))/gamma)
    L=Xo*tanh(gamma*length)/w0
    L1=L1*L/(L-L1)
    L1=L1*Lj/(Lj-L1/n/n)
    L2=L2*Xo*tanh(gamma*length)/(Xo*tanh(gamma*length)-2*w0*L2)
    return (length,C1,C2,L1,L2,J)

if __name__ == "__main__":
    #print ButterworthFilterPrototype(5)
    from pylab import *
    from network import *
    a=0.016
    a1=0.008
    b=0.0005
    #b=0.001524
    er=3.38
    c2_c1=2
    n=1.6       #symmetric
    X=0.4e-9    #symmetric

    n=1.6       #symmetric
    X=1.2e-9    #symmetric

    #n=2.1       #asymmetric
    #X=0.16e-9   #asymmetric
    loss=1 #plot loss if 1, plot isolation if 0
    (length,c1,c2,l1, l2, j)=EvanescentWGFilter_3(ButterworthFilterPrototype(3),n,X,a,a1, b,er, 8.5e9,0.25,c2_c1)
    print("length= ",length)
    print("C1= ",c1)
    print("C2= ",c2)
    print("L1= ",l1)
    print("L2= ",l2)
    if l1<0:
        l1=10000
    if l2<0:
        l2=10000
    sonuc=[]
    #length=0.0042
    #l2=0.1145e-9
    #c2=4.847e-12
    l2i=0.042e-9
    r2i=0.26
    l1i=0.16e-9
    r1i=0.98

    for frek in linspace(6e9,10e9,101):
        networks=[]
        networks.append(shunt_z(2.0j*pi*frek*X))
        networks.append(transformer(n))
        if loss:
            networks.append(shunt_z(-1.0j/2./pi/frek/c1))
            networks.append(shunt_z(2.0j*pi*frek*l1))
        else:
            networks.append(shunt_z(2.0j*pi*frek*l1i+r1i))

        networks.append(EWG_ABCD(a1,b,er,length,frek))
        #networks.append(EWG_inv(a1,b,er,length,frek))
        #networks.append(jinv(j))
        #networks.append(shunt_z(-1.0j/j))
        #networks.append(series_z(1.0j/j))
        #networks.append(shunt_z(-1.0j/j))
        if loss:
            networks.append(shunt_z(-1.0j/2./pi/frek/c2))
            networks.append(shunt_z(2.0j*pi*frek*l2))
        else:
            networks.append(shunt_z(2.0j*pi*frek*l2i+r2i))
        networks.append(EWG_ABCD(a1,b,er,length,frek))
        #networks.append(EWG_inv(a1,b,er,length,frek))
        #networks.append(jinv(j))
        #networks.append(shunt_z(-1.0j/j))
        #networks.append(series_z(1.0j/j))
        #networks.append(shunt_z(-1.0j/j))
        if loss:
            networks.append(shunt_z(-1.0j/2./pi/frek/c1))
            networks.append(shunt_z(2.0j*pi*frek*l1))
        else:
            networks.append(shunt_z(2.0j*pi*frek*l1i+r1i))

        networks.append(transformer((1./n)))
        networks.append(shunt_z(2.0j*pi*frek*X))
        abcd=cascade_networks(networks)
        sonuc.append(abcd2s(abcd,Z_WG_TE10(er,a,b,frek))[0,1])
    #sonuc.append(abcd2s(abcd,Z_WG_TE10(er,a,b,8.5e9))[0,1])
    plot(linspace(6e9,10e9,101),20.0*log10(abs(array(sonuc))))
    grid()
    xlabel("Frequency")
    if loss:
        ylabel("Insertion Loss")
    else:
        ylabel("Isolation")
    show()
    print(filter_with_inverter2(ButterworthFilterPrototype(3),16,8.5e9,0.125,[0.1e-12,0.5e-12,0.1e-12]))
