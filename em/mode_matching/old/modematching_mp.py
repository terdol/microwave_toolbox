# -*- coding:utf-8 -*-
import numpy as np
from numpy.lib.scimath import sqrt as csqrt
from scipy.integrate import dblquad
import itertools
import multiprocessing
import time
current=time.time()

mu0=4*np.pi*1e-7
eps0=8.854187817e-12
c0=299792458
pi = np.pi

def Ex(x,y,xc,yc,m,n,a,b,eps,freq,mode=1):
    """ Mode=1 for TE, 0 for TM """
    w = 2*np.pi*freq
    kc=np.sqrt((n*pi/b)**2+(m*pi/a)**2)
    k2 = w**2*eps/c0**2-kc**2
    if k2>0:
        kz = np.sqrt(k2)
    else:
        kz= -1j*np.sqrt(-k2)
    if mode:
        return 1j*w*mu0/kc**2*(n*pi/b)*np.cos(m*pi*(x-xc)/a)*np.sin(n*pi*(y-yc)/b)
    else:
        return -1j*kz/kc**2*(m*pi/a)*np.cos(m*pi*(x-xc)/a)*np.sin(n*pi*(y-yc)/b)
        
def Ey(x,y,xc,yc,m,n,a,b,eps,freq,mode=1):
    """ Mode=1 for TE, 0 for TM """
    w = 2*np.pi*freq
    kc=np.sqrt((n*pi/b)**2+(m*pi/a)**2)
    k2 = w**2*eps/c0**2-kc**2
    if k2>0:
        kz = np.sqrt(k2)
    else:
        kz= -1j*np.sqrt(-k2)
    if mode:
        return -1j*w*mu0/kc**2*(m*pi/a)*np.sin(m*pi*(x-xc)/a)*np.cos(n*pi*(y-yc)/b)
    else:
        return -1j*kz/kc**2*(n*pi/b)*np.sin(m*pi*(x-xc)/a)*np.cos(n*pi*(y-yc)/b)

def Hx(x,y,xc,yc,m,n,a,b,eps,freq,mode=1):
    """ Mode=1 for TE, 0 for TM """
    w = 2*np.pi*freq
    kc=np.sqrt((n*pi/b)**2+(m*pi/a)**2)
    k2 = w**2*eps/c0**2-kc**2
    if k2>0:
        kz = np.sqrt(k2)
    else:
        kz= -1j*np.sqrt(-k2)
    if mode:
        return 1j*kz/kc**2*(m*pi/a)*np.sin(m*pi*(x-xc)/a)*np.cos(n*pi*(y-yc)/b)
    else:
        return 1j*w*eps0/kc**2*(n*pi/b)*np.sin(m*pi*(x-xc)/a)*np.cos(n*pi*(y-yc)/b)
        
def Hy(x,y,xc,yc,m,n,a,b,eps,freq,mode=1):
    """ Mode=1 for TE, 0 for TM """
    w = 2*np.pi*freq
    kc=np.sqrt((n*pi/b)**2+(m*pi/a)**2)
    k2 = w**2*eps/c0**2-kc**2
    if k2>0:
        kz = np.sqrt(k2)
    else:
        kz= -1j*np.sqrt(-k2)
    if mode:
        return 1j*kz/kc**2*(n*pi/b)*np.cos(m*pi*(x-xc)/a)*np.sin(n*pi*(y-yc)/b)
    else:
        return -1j*w*eps*eps0/kc**2*(m*pi/a)*np.cos(m*pi*(x-xc)/a)*np.sin(n*pi*(y-yc)/b)


xc1=0
yc1=0
xc2=0
yc2=1.0e-3
# yc2=0
a1=3.1e-3
b1=1.55e-3
a2=3.1e-3
b2=0.55e-3
eps=1
freq=77e9

freq_cutoff=300e9
# Maximum mode numbers that are taken into account
n1=int(2*pi*freq_cutoff/c0*b1/pi)
m1=int(2*pi*freq_cutoff/c0*a1/pi)
n2=int(2*pi*freq_cutoff/c0*b2/pi)
m2=int(2*pi*freq_cutoff/c0*a2/pi)


# List of tuples (i,j,mode)
# indice+1 for each mode is also its total mode number
TEmodes1= list(itertools.product(list(range(m1+1)),list(range(n1+1)),[1]))
TEmodes2= list(itertools.product(list(range(m2+1)),list(range(n2+1)),[1]))
TMmodes1= list(itertools.product(list(range(1,m1+1)),list(range(1,n1+1)),[0]))
TMmodes2= list(itertools.product(list(range(1,m2+1)),list(range(1,n2+1)),[0]))
TEmodes1.pop(TEmodes1.index((0,0,1)))
TEmodes2.pop(TEmodes2.index((0,0,1)))
modes1 = TEmodes1 + TMmodes1
modes2 = TEmodes2 + TMmodes2

print(modes1)
print(modes2)
        
Nw = len(modes1)
Ns = len(modes2)

def eh(x,y,params):
    return Ex(x,y,*params)*Hy(x,y,*params)-Ey(x,y,*params)*Hx(x,y,*params)
Qw = np.matrix(np.zeros((Nw,Nw),dtype=complex))
for i in range(Nw):
    m,n,mode = modes1[i]
    params=(xc1,yc1,m,n,a1,b1,eps,freq,mode)
    py, err = dblquad(lambda x,y:np.imag(eh(x,y,params)), yc1, yc1+b1,
                       lambda y: xc1,
                       lambda y: a1+xc1)
    px, err = dblquad(lambda x,y:np.real(eh(x,y,params)), yc1, yc1+b1,
                       lambda y: xc1,
                       lambda y: a1+xc1)
    Qw[i,i] = px+py*1j



print(time.time()-current)
current=time.time()

Qs = np.matrix(np.zeros((Ns,Ns),dtype=complex))
for i in range(Ns):
    m,n,mode = modes2[i]
    params=(xc2,yc2,m,n,a2,b2,eps,freq,mode)
    py, err = dblquad(lambda x,y:np.imag(eh(x,y,params)), yc2, yc2+b2,
                       lambda y: xc2,
                       lambda y: a2+xc2)
    px, err = dblquad(lambda x,y:np.real(eh(x,y,params)), yc2, yc2+b2,
                       lambda y: xc2,
                       lambda y: a2+xc2)
    Qs[i,i] = px+py*1j


print(time.time()-current)
current=time.time()


modes12= list(itertools.product(modes1, modes2))
def exh(x,y,par1,par2):
    return Ex(x,y,*par2)*Hy(x,y,*par1)-Ey(x,y,*par2)*Hx(x,y,*par1)

XXR = np.matrix(np.zeros((Ns,Nw),dtype=complex))
XXI = np.matrix(np.zeros((Ns,Nw),dtype=complex))
def calc_real():
    print("real started")
    for i in range(len(modes2)):
        m2,n2,mode2 = modes2[i]
        params2=(xc2,yc2,m2,n2,a2,b2,eps,freq,mode2) 
        for j in range(len(modes1)):
            m1,n1,mode1 = modes1[j]
            params1=(xc1,yc1,m1,n1,a1,b1,eps,freq,mode1)
            px, err = dblquad(lambda x,y:np.real(exh(x,y,params1,params2)), yc2, yc2+b2,
                               lambda y: xc2,
                               lambda y: a2+xc2,epsabs=1e-3,epsrel=1e-3)
            XXR[i,j] = px
    print("real finished")
    

def calc_imag():
    print("imag started")
    for i in range(len(modes2)):
        m2,n2,mode2 = modes2[i]
        params2=(xc2,yc2,m2,n2,a2,b2,eps,freq,mode2) 
        for j in range(len(modes1)):
            m1,n1,mode1 = modes1[j]
            params1=(xc1,yc1,m1,n1,a1,b1,eps,freq,mode1)
            py, err = dblquad(lambda x,y:np.imag(exh(x,y,params1,params2)), yc2, yc2+b2,
                               lambda y: xc2,
                               lambda y: a2+xc2,epsabs=1e-3,epsrel=1e-3)
            XXI[i,j] = py*1j
    print("imag finished")


proses1 = multiprocessing.Process(None,calc_real,args=())
proses1.start()
time.sleep(2)
proses2 = multiprocessing.Process(None,calc_imag,args=())
proses2.start()
proses1.join()
proses2.join()

XX= XXR + XXI
print(time.time()-current)
current=time.time()

Iw = np.matrix(np.eye(Nw))
Is = np.matrix(np.eye(Ns))
SP = np.matrix(np.zeros((Ns+Nw,Ns+Nw),dtype=complex))

F = 2*(Qs+XX*Qw.I*XX.T).I

S11 = Qw.I*XX.T*F*XX-Iw
S12 = Qw.I*XX.T*F*Qs
S21 = F*XX
S22 = F*Qs-Is
print(np.shape(S11))
print(np.shape(S12))
print(np.shape(S21))
print(np.shape(S22))
SP[:Nw,:Nw] = S11
SP[:Nw,Nw:] = S12
SP[Nw:,:Nw] = S21
SP[Nw:,Nw:] = S22

print(time.time()-current)
current=time.time()

SSP = SP*SP
cc1=modes1.index((1,0,1))
cc2=modes2.index((1,0,1))
print(SP[cc1,cc1])
print(SP[Nw+cc2,cc1])