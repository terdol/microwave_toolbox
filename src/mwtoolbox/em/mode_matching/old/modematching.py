# -*- coding:utf-8 -*-
import numpy as np
from numpy.lib.scimath import sqrt as csqrt
from scipy.integrate import dblquad
import itertools

mu0=4*np.pi*1e-7
eps0=8.85e-12
c0=3e8
pi = np.pi

def Ex(x,y,xc,yc,m,n,a,b,eps,freq,mode=1):
    """ Mode=1 for TE, 0 for TM """
    w = 2*np.pi*freq
    kc=np.sqrt((n*pi/b)**2+(m*pi/a)**2)
    kz = csqrt(w**2*eps/c0**2-kc**2)
    if mode:
        return 1j*w*mu0/kc**2*(n*pi/b)*np.cos(m*pi*(x-xc)/a)*np.sin(n*pi*(y-yc)/b)
    else:
        return -1j*kz/kc**2*(m*pi/a)*np.cos(m*pi*(x-xc)/a)*np.sin(n*pi*(y-yc)/b)
        
def Ey(x,y,xc,yc,m,n,a,b,eps,freq,mode=1):
    """ Mode=1 for TE, 0 for TM """
    w = 2*np.pi*freq
    kc=np.sqrt((n*pi/b)**2+(m*pi/a)**2)
    kz = csqrt(w**2*eps/c0**2-kc**2)
    if mode:
        return -1j*w*mu0/kc**2*(m*pi/a)*np.sin(m*pi*(x-xc)/a)*np.cos(n*pi*(y-yc)/b)
    else:
        return -1j*kz/kc**2*(n*pi/b)*np.sin(m*pi*(x-xc)/a)*np.cos(n*pi*(y-yc)/b)

def Hx(x,y,xc,yc,m,n,a,b,eps,freq,mode=1):
    """ Mode=1 for TE, 0 for TM """
    w = 2*np.pi*freq
    kc=np.sqrt((n*pi/b)**2+(m*pi/a)**2)
    kz = csqrt(w**2*eps/c0**2-kc**2)
    if mode:
        return 1j*kz/kc**2*(m*pi/a)*np.sin(m*pi*(x-xc)/a)*np.cos(n*pi*(y-yc)/b)
    else:
        return 1j*w*eps0/kc**2*(n*pi/b)*np.sin(m*pi*(x-xc)/a)*np.cos(n*pi*(y-yc)/b)
        
def Hy(x,y,xc,yc,m,n,a,b,eps,freq,mode=1):
    """ Mode=1 for TE, 0 for TM """
    w = 2*np.pi*freq
    kc=np.sqrt((n*pi/b)**2+(m*pi/a)**2)
    kz = csqrt(w**2*eps/c0**2-kc**2)
    if mode:
        return 1j*kz/kc**2*(n*pi/b)*np.cos(m*pi*(x-xc)/a)*np.sin(n*pi*(y-yc)/b)
    else:
        return -1j*w*eps*eps0/kc**2*(m*pi/a)*np.cos(m*pi*(x-xc)/a)*np.sin(n*pi*(y-yc)/b)
        


# def modenumber(i,j,m,mode=1):
    # """ i: x-mode number (starts from 0 for TE, 1 for TM)
        # j: y-mode number (starts from 0 for TE, 1 for TM)
        # m: maximum x-mode number
        # mode: 1 if TE, 0 if TM
    # """
    # if mode:
        # return (j)*(m+1)+i
    # else:
        # return (j)*(m)+i

# def row_column_X(i,j,mode):
    # """ Ns x Nw """
    # if mode:
        # r = modenumber(i,j,m2,mode)
        # c = modenumber(i,j,m1,mode)
    # else:
        # r = modenumber(i,j,m2,mode)
        # c = modenumber(i,j,m1,mode)
        
# def row_column_Xt(i,j,mode):
    # if mode:
        # k = modenumber(i,j,m1,mode)
    # else:
        # k = modenumber(i,j,m1,mode)
        # k = k+Nw1

xc=0
yc=0
m1=4
n1=2
m2=4
n2=2
a=3.1e-3
b=1.55e-3
a1=3.1e-3
b1=1.55e-3
a2=3.1e-3
b2=1.55e-3
eps=1
freq=77e9
# Q-Function
params1=(xc,yc,m1,n1,a,b,eps,freq,1)
params2=(xc,yc,m2,n2,a,b,eps,freq,1)
ans, err = dblquad(lambda x,y:np.imag(Ex(x,y,*params1)*Hy(x,y,*params2)-Ey(x,y,*params1)*Hx(x,y,*params2)), yc, yc+b,
                   lambda y: xc,
                   lambda y: a+xc)
                   
print(ans)

freq_cutoff=500e9
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
# number of modes for TE
# Nw1=(n1+1)*(m1+1)-1
# Ns1=(n2+1)*(m2+1)-1
# number of modes for TM
# Nw2=(n1+1)*m1
# Ns2=(n2+1)*m2

# Nw = Nw1 + Nw2
# Ns = Ns1 + Ns2

Nw = len(modes1)
Ns = len(modes2)

Qw = np.zeros((Nw,Nw),dtype=complex)
for i in range(Nw):
    m,n,mode = modes1[i]
    params=(xc1,yc1,m,n,a1,b1,eps,freq,mode)
    y, err = dblquad(lambda x,y:np.imag(Ex(x,y,*params)*Hy(x,y,*params)-Ey(x,y,*params)*Hx(x,y,*params)), yc1, yc1+b1,
                       lambda y: xc1,
                       lambda y: a1+xc1)
    x, err = dblquad(lambda x,y:np.real(Ex(x,y,*params)*Hy(x,y,*params)-Ey(x,y,*params)*Hx(x,y,*params)), yc1, yc1+b1,
                       lambda y: xc1,
                       lambda y: a1+xc1)
    Qw[i,i] = x+y*1j
    

Qs = np.zeros((Ns,Ns),dtype=complex)
for i in range(Ns):
    m,n,mode = modes2[i]
    params=(xc2,yc2,m,n,a2,b2,eps,freq,mode)
    y, err = dblquad(lambda x,y:np.imag(Ex(x,y,*params)*Hy(x,y,*params)-Ey(x,y,*params)*Hx(x,y,*params)), yc2, yc2+b2,
                       lambda y: xc2,
                       lambda y: a2+xc2)
    x, err = dblquad(lambda x,y:np.real(Ex(x,y,*params)*Hy(x,y,*params)-Ey(x,y,*params)*Hx(x,y,*params)), yc2, yc2+b2,
                       lambda y: xc2,
                       lambda y: a2+xc2)
    Qw[i,i] = x+y*1j
    
XX = np.zeros((Ns,Nw),dtype=complex)
modes12= list(itertools.product(modes1, modes2))
for i in range(range(len(modes2))):
    for j in range(range(len(modes1))):
        m1,n1,mode1 = modes1[j]
        m2,n2,mode2 = modes2[i]
        params1=(xc1,yc1,m1,n1,a1,b1,eps,freq,mode1)
        params2=(xc2,yc2,m2,n2,a2,b2,eps,freq,mode2)
        def exh(x,y):
            return Ex(x,y,*params2)*Hy(x,y,*params1)-Ey(x,y,*params2)*Hx(x,y,*params1)
        y, err = dblquad(lambda x,y:np.imag(exh(x,y)), yc2, yc2+b2,
                           lambda y: xc2,
                           lambda y: a2+xc2)
        x, err = dblquad(lambda x,y:np.real(exh(x,y)), yc2, yc2+b2,
                           lambda y: xc2,
                           lambda y: a2+xc2)
        XX[i,j] = x+y*1j

Iw = np.eye(Nw)
Is = np.eye(Ns)
SP = np.zeros((Ns*Nw,Ns*Nw),dtype=complex)
F = 2*(Qs+XX*Qw.I*XX.T).I
S11 = Qw.I*XX.T*F*XX-Iw
S12 = Qw.I*XX.T*F*Qs
S21 = F*XX
S22 = F*Qs-Is

SP[:Nw,:Nw] = S11
SP[:Nw,Nw:] = S12
SP[Nw:,:Nw] = S21
SP[Nw:,Nw:] = S22
