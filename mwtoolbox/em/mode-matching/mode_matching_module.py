# -*- coding:utf-8 -*-
# Mode-Matching code to analyse waveguide junction
# Reference: Computer Aided Design of Waveguide Devices By Mode-Matching Method
import numpy as np
from numpy.lib.scimath import sqrt as csqrt
from scipy.integrate import dblquad
import itertools
import time
current=time.time()

mu0=4*np.pi*1e-7
eps0=8.854187817e-12
c0=299792458
pi = np.pi

def isqrt(k2):
    if k2>0:
        kz = np.sqrt(k2)
    else:
        kz= -1j*np.sqrt(-k2)
    return kz

def Ex(x,y,xc,yc,m,n,a,b,eps,freq,mode=1):
    """ Mode=1 for TE, 0 for TM """
    w = 2*pi*freq
    kc = np.sqrt((n*pi/b)**2+(m*pi/a)**2)
    k2 = w**2*eps/c0**2-kc**2
    kz = isqrt(k2)
    if mode:
        return 1j*w*mu0/kc**2*(n*pi/b)*np.cos(m*pi*(x-xc)/a)*np.sin(n*pi*(y-yc)/b)
    else:
        return -1j*kz/kc**2*(m*pi/a)*np.cos(m*pi*(x-xc)/a)*np.sin(n*pi*(y-yc)/b)

def Ey(x,y,xc,yc,m,n,a,b,eps,freq,mode=1):
    """ Mode=1 for TE, 0 for TM """
    w  = 2*pi*freq
    kc = np.sqrt((n*pi/b)**2+(m*pi/a)**2)
    k2 = w**2*eps/c0**2-kc**2
    kz = isqrt(k2)
    if mode:
        return -1j*w*mu0/kc**2*(m*pi/a)*np.sin(m*pi*(x-xc)/a)*np.cos(n*pi*(y-yc)/b)
    else:
        return -1j*kz/kc**2*(n*pi/b)*np.sin(m*pi*(x-xc)/a)*np.cos(n*pi*(y-yc)/b)

def Hx(x,y,xc,yc,m,n,a,b,eps,freq,mode=1):
    """ Mode=1 for TE, 0 for TM """
    w  = 2*pi*freq
    kc = np.sqrt((n*pi/b)**2+(m*pi/a)**2)
    k2 = w**2*eps/c0**2-kc**2
    kz = isqrt(k2)
    if mode:
        return 1j*kz/kc**2*(m*pi/a)*np.sin(m*pi*(x-xc)/a)*np.cos(n*pi*(y-yc)/b)
    else:
        return 1j*w*eps0*eps/kc**2*(n*pi/b)*np.sin(m*pi*(x-xc)/a)*np.cos(n*pi*(y-yc)/b)

def Hy(x,y,xc,yc,m,n,a,b,eps,freq,mode=1):
    """ Mode=1 for TE, 0 for TM """
    w  = 2*pi*freq
    kc = np.sqrt((n*pi/b)**2+(m*pi/a)**2)
    k2 = w**2*eps/c0**2-kc**2
    kz = isqrt(k2)
    if mode:
        return 1j*kz/kc**2*(n*pi/b)*np.cos(m*pi*(x-xc)/a)*np.sin(n*pi*(y-yc)/b)
    else:
        return -1j*w*eps*eps0/kc**2*(m*pi/a)*np.cos(m*pi*(x-xc)/a)*np.sin(n*pi*(y-yc)/b)

def eh(x,y,params):
    return Ex(x,y,*params)*Hy(x,y,*params)-Ey(x,y,*params)*Hx(x,y,*params)

def createmesh(xcs,ycs,xcf,ycf,Nx,Ny):
    dx=(xcf-xcs)/Nx
    dy=(ycf-ycs)/Ny
    xp = np.linspace(xcs+dx/2,xcf-dx/2,Nx)
    yp = np.linspace(ycs+dy/2,ycf-dy/2,Ny)
    xp_1, yp_1 = np.meshgrid(xp,yp)
    dS = dx*dy
    return xp_1, yp_1, dS

def exh(x,y,par1,par2):
    return Ex(x,y,*par2)*Hy(x,y,*par1)-Ey(x,y,*par2)*Hx(x,y,*par1)

def sp_wg(a,b,eps,L,modes):
    temp=[np.exp(-L*1j*isqrt(w**2*eps/c0**2-((n*pi/b)**2+(m*pi/a)**2))) for m,n,k in modes]
    return np.matrix(np.diag(temp))

class MM_RWG:
    def __init__(self,wg1, wg2, aperture=None):
        self.xc1=wg1[0]
        self.yc1=wg1[1]
        self.xc2=wg2[0]
        self.yc2=wg2[1]
        self.a1=wg1[2]
        self.b1=wg1[3]
        self.a2=wg2[2]
        self.b2=wg2[3]
        self.eps1=wg1[4]
        self.eps2=wg2[4]
        self.aperture = aperture

    def simulate(self, freqs):
        xc1 = self.xc1
        yc1 = self.yc1
        xc2 = self.xc2
        yc2 = self.yc2
        a1 = self.a1
        b1 = self.b1
        a2 = self.a2
        b2 = self.b2
        eps1 = self.eps1
        eps2 = self.eps2
        Nx=40
        Ny=20
        freq_cutoff=np.max(freqs)*5
        # Boundaries of aperture
        if self.aperture:
            xt2 = self.aperture[1]
            xt1 = self.aperture[0]
            yt2 = self.aperture[3]
            yt1 = self.aperture[2]
        else:
            xt2 = np.min([xc1+a1,xc2+a2])
            xt1 = np.max([xc1,xc2])
            yt2 = np.min([yc1+b1,yc2+b2])
            yt1 = np.max([yc1,yc2])
        print(xt1,xt2,yt1,yt2)
        # Maximum mode numbers that are taken into account
        n1=int(2*pi*freq_cutoff/c0*b1/pi)
        m1=int(2*pi*freq_cutoff/c0*a1/pi)
        n2=int(2*pi*freq_cutoff/c0*b2/pi)
        m2=int(2*pi*freq_cutoff/c0*a2/pi)
        n3=int(2*pi*freq_cutoff/c0*(yt2-yt1)/pi)
        m3=int(2*pi*freq_cutoff/c0*(xt2-xt1)/pi)

        # These are list of modes, each element is a 3-tuple (m,n,k)
        #   m: wavenumber in x-axis
        #   n: wavenumber in y-axis
        #   k: type of mode (1:TE, 0:TM)
        # modes1: Mode expansion at WG-1
        # modes2: Mode expansion at WG-2
        # modes3: Mode expansion at aperture
        TEmodes1= list(itertools.product(list(range(m1+1)),list(range(n1+1)),[1]))
        TEmodes2= list(itertools.product(list(range(m2+1)),list(range(n2+1)),[1]))
        TEmodes3= list(itertools.product(list(range(m3+1)),list(range(n3+1)),[1]))
        TMmodes1= list(itertools.product(list(range(1,m1+1)),list(range(1,n1+1)),[0]))
        TMmodes2= list(itertools.product(list(range(1,m2+1)),list(range(1,n2+1)),[0]))
        TMmodes3= list(itertools.product(list(range(1,m3+1)),list(range(1,n3+1)),[0]))
        TEmodes1.pop(TEmodes1.index((0,0,1))) # remove undefined modes
        TEmodes2.pop(TEmodes2.index((0,0,1))) # remove undefined modes
        TEmodes3.pop(TEmodes3.index((0,0,1))) # remove undefined modes
        modes1 = TEmodes1 + TMmodes1
        modes2 = TEmodes2 + TMmodes2
        modes3 = TEmodes3 + TMmodes3
        modes13= list(itertools.product(modes1, modes3))
        modes23= list(itertools.product(modes2, modes3))
        Nw1 = len(modes1)
        Nw2 = len(modes2)
        Ns  = len(modes3)

        xp_1, yp_1, dS1 = createmesh(xc1,yc1,xc1+a1,yc1+b1,Nx,Ny)
        xp_2, yp_2, dS2 = createmesh(xc2,yc2,xc2+a2,yc2+b2,Nx,Ny)
        xp_3, yp_3, dS3 = createmesh(xt1,yt1,xt2,yt2,Nx,Ny)

        Iw1 = np.matrix(np.eye(Nw1))
        Iw2 = np.matrix(np.eye(Nw2))
        Is  = np.matrix(np.eye(Ns))
        SP  = np.matrix(np.zeros((Nw1+Nw2,Nw1+Nw2),dtype=complex))
        Qw1 = np.matrix(np.zeros((Nw1,Nw1),dtype=complex))
        Qw2 = np.matrix(np.zeros((Nw2,Nw2),dtype=complex))
        Qs  = np.matrix(np.zeros((Ns,Ns),dtype=complex))

        SPARAMS = []
        for freq in freqs:
            print(freq)
            for i in range(Nw1):
                m,n,mode = modes1[i]
                params=(xc1,yc1,m,n,a1,b1,eps1,freq,mode)
                Qw1[i,i] = np.sum(eh(xp_1,yp_1,params)*dS1)

            for i in range(Nw2):
                m,n,mode = modes2[i]
                params=(xc2,yc2,m,n,a2,b2,eps2,freq,mode)
                Qw2[i,i] = np.sum(eh(xp_2,yp_2,params)*dS2)

            if xt2<xt1 or yt2<yt1:
                Xw1=0
                Xw2=0
                print("hh")
            else:
                Xw1 = np.matrix(np.zeros((Ns,Nw1),dtype=complex))
                for i in range(len(modes3)):
                    m3,n3,mode3 = modes3[i]
                    params2=(xt1,yt1,m3,n3,xt2-xt1,yt2-yt1,0.5*(eps1+eps2),freq,mode3)
                    for j in range(len(modes1)):
                        m1,n1,mode1 = modes1[j]
                        params1=(xc1,yc1,m1,n1,a1,b1,eps1,freq,mode1)
                        Xw1[i,j] = np.sum(exh(xp_3,yp_3,params1,params2)*dS3)
                Xw2 = np.matrix(np.zeros((Ns,Nw2),dtype=complex))
                for i in range(len(modes3)):
                    m3,n3,mode3 = modes3[i]
                    params2=(xt1,yt1,m3,n3,xt2-xt1,yt2-yt1,0.5*(eps1+eps2),freq,mode3)
                    for j in range(len(modes2)):
                        m2,n2,mode2 = modes2[j]
                        params1=(xc2,yc2,m2,n2,a2,b2,eps2,freq,mode2)
                        Xw2[i,j] = np.sum(exh(xp_3,yp_3,params1,params2)*dS3)

            F = 2*(Xw1*Qw1.I*Xw1.T+Xw2*Qw2.I*Xw2.T).I
            S11 = Qw1.I*Xw1.T*F*Xw1-Iw1
            S12 = Qw1.I*Xw1.T*F*Xw2
            S21 = Qw2.I*Xw2.T*F*Xw1
            S22 = Qw2.I*Xw2.T*F*Xw2-Iw2
            SP[:Nw1,:Nw1] = S11
            SP[:Nw1,Nw1:] = S12
            SP[Nw1:,:Nw1] = S21
            SP[Nw1:,Nw1:] = S22
            SPARAMS.append(SP)
        return SPARAMS, modes1, modes2

if __name__=="__main__":
    xc1=0
    yc1=0
    xc2=0
    yc2=0
    a1=3.1e-3
    b1=1.55e-3
    a2=3.1e-3
    # b2=0.55e-3
    b2=1.55e-3
    eps1=1
    eps2=1
    aperture1=(xc1,yc1,a1,b1,eps1)
    aperture2=(xc2,yc2,a2,b2,eps2)
    aperture3=(1.05e-3,2.05e-3,0,1.55e-3)
    # freqs=np.linspace([65e9,90e9,26])
    freqs=[77e9]
    solver=MM_RWG(aperture1,aperture2,aperture3)
    Sparams, modes1, modes2 = solver.simulate(freqs)
    cc1=modes1.index((1,0,1))
    print(cc1)
    print(Sparams[0][cc1,cc1])
    # import matplotlib.pyplot as plt
    # plt.plot(freqs, 20*np.log10(np.abs(np.array(RL))))
    # plt.grid()
    # plt.show()
