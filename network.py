#-*-coding:utf-8-*-
"""
Network Parameters
"""
from __future__ import print_function
from __future__ import division

from builtins import range
import numpy as np
import operator as op
from numpy.lib.scimath import sqrt as csqrt
from functools import reduce
#ABCD parameters
def shZ(Z): 
    """
    ABCD parameters of shunt impedance
    """
    return np.matrix([[1., 0], [(1./Z), 1.]])
def seZ(Z): 
    """
    ABCD parameters of series impedance
    """
    return np.matrix([[1., Z], [0, 1.]])
def jinv(J): 
    """    
    ABCD parameters of J - inverter
    """
    return np.matrix([[0, 1.0j/J], [1.0j * J, 0]])
def jinv_lumped(X): 
    """
    ABCD parameters of J - inverter produced by 3 inductors in Tee form.
    """
    return CascadeNetworks([shZ( - X), seZ(X), shZ( - X)])
def kinv(K): 
    """
    ABCD parameters of k - inverter
    """
    return np.matrix([[0, 1.0j * K], [1.0j/K, 0]])
def tline(Zo, theta): 
    """
    ABCD parameters of ideal transmission line,  theta = radian
    """
    return np.matrix([[np.cos(theta), 1.0j * Zo * np.sin(theta)], [1.0j/Zo * np.sin(theta), np.cos(theta)]])
def transformer(N): 
    """
    ABCD parameters of ideal transformer (1:N)
    """
    return np.matrix([[1./N, 0], [0, N]])
def t_network(Zs1, Zp, Zs2): 
    """
    ABCD parameters of Tee network
    """
    return np.matrix([[1 + Zs1/Zp, Zs1 + Zs2 + Zs1 * Zs2/Zp], [1./Zp, 1. + Zs2/Zp]])
def p_network(Zp1, Zs, Zp2): 
    """
    ABCD parameters of Pi network
    """
    return np.matrix([[1 + Zs/Zp2, Zs], [1./Zp1 + 1./Zp2 + Zs/Zp1/Zp2, 1 + Zs/Zp1]])
    
def ABCD2Y(M): 
    """
    ABCD parameters to Y - Parameters Conversion
    """
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    if b == 0:
        print("Undefined Y Parameter")
        return
    return np.matrix([[d/b, (b * c - a * d)/b], [ -1./b, a/b]])
    
def Y2ABCD(M): 
    """
    Y-Parameters to ABCD parameters Conversion
    """
    y11, y12, y21, y22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    if y21 == 0:
        print("Undefined Y Parameter")
        return
    return np.matrix([[ -y22/y21,  -1./y21], [(y12 * y21 - y11 * y22)/y21,  -y11/y21]])
    
def ABCD2Z(M): 
    """
    ABCD parameters to Z - Parameters Conversion
    """
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    if c == 0:
        print("Undefined Z Parameter")
        return
    return np.matrix([[a/c, ( - b * c + a * d)/c], [1./c, d/c]])
    
def Z2ABCD(M): 
    """
    Z - Parameters to ABCD parameters Conversion
    """
    z11, z12, z21, z22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    if z21 == 0:
        print("Undefined Y Parameter")
        return
    return np.matrix([[z11/z21, (z11 * z22 - z21 * z12)/z21], [1./z21, z22/z21]])
    
def ABCD2S(M, Zo): 
    """
    ABCD parameters to S - Parameters Conversion
    """
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    s11 = ((a + b/Zo - c * Zo - d)/(a + b/Zo + c * Zo + d))
    s12 = 2. * (a * d - b * c)/(a + b/Zo + c * Zo + d)
    s21 = (2./(a + b/Zo + c * Zo + d))
    s22 = (( - a + b/Zo - c * Zo + d)/(a + b/Zo + c * Zo + d))
    return np.matrix([[s11, s12], [s21, s22]])

def S2ABCD(M, Z): 
    """
    ABCD parameters to S-Parameters Conversion
    """
    Zo1, Zo2 = tuple(Z)
    s11, s12, s21, s22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    a = ((Zo1.conjugate() + s11 * Zo1) * (1. - s22) + s12 * s21 * Zo1)/(2. * s21 * csqrt(Zo1.real * Zo2.real))
    b = ((Zo1.conjugate() + s11 * Zo1) * (Zo2.conjugate() + s22 * Zo2) - s12 * s21 * Zo1 * Zo2)/(2. * s21 * csqrt(Zo1.real * Zo2.real))
    c = ((1. - s11) * (1. - s22) - s12 * s21)/(2. * s21 * csqrt(Zo1.real * Zo2.real))
    d = ((1. - s11) * (Zo2.conjugate() + s22 * Zo2) + s12 * s21 * Zo2)/(2. * s21 * csqrt(Zo1.real * Zo2.real))
    return np.matrix([[a, b], [c, d]])    
    
def ABCD2T(M, Z): 
    """
    ABCD parameters to T - Parameters Conversion
    """
    Zo1, Zo2 = tuple(Z)
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    t11 = (a * Zo2 + b + c * Zo1 * Zo2 + d * Zo1)/2./csqrt(Zo1.real * Zo2.real)
    t21 = (a * Zo2 + b + c * Zo1.conjugate() * Zo2 - d * Zo1.conjugate())/2./csqrt(Zo1.real * Zo2.real)
    t12 = (a * Zo2.conjugate() - b + c * Zo1 * Zo2.conjugate() - d * Zo1)/2./csqrt(Zo1.real * Zo2.real)
    t22 = (a * Zo2.conjugate() - b - c * Zo1.conjugate() * Zo2.conjugate() + d * Zo1.conjugate())/2./csqrt(Zo1.real * Zo2.real)
    return np.matrix([[t11, t12], [t21, t22]])
    
def ABCD_ChangePorts(M): 
    """
    Switching ports of ABCD parameters 
    """
    M = M.I
    np.matrix([[M[0, 0],  - M[0, 1]], [ - M[1, 0], M[1, 1]]])

def T2ABCD(M, Z): 
    """
    T-parameters to ABCD parameters Conversion
    """
    Zo1, Zo2 = tuple(Z)
    t11, t12, t21, t22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    c = (t11 + t12 - t21 - t22)/2./csqrt(Zo1.real * Zo2.real)
    d = (Zo2.conjugate() * (t11 - t21) - Zo2 * (t12 - t22))/2./csqrt(Zo1.real * Zo2.real)
    a = (Zo1.conjugate() * (t11 + t12) + Zo1 * (t21 + t22))/2./csqrt(Zo1.real * Zo2.real)
    b = (Zo2.conjugate() * (t11 * Zo1.conjugate() + t21 * Zo1) - Zo2 * (t12 * Zo1.conjugate() + t22 * Zo1))/2./csqrt(Zo1.real * Zo2.real)
    return np.matrix([[a, b], [c, d]])
    
def CascadeNetworks(networks): 
    """
    Cascading 2-port Networks,  input and output is ABCD matrices of networks
    """
    return reduce(op.mul, networks)

def ParallelNetworks(networks):
    """
    Paralleling 2-port Networks,  input and output  is ABCD matrices of networks
    """
    ymatrices = [ABCD2Y(M) for M in networks]
    return Y2ABCD(reduce(op.add, ymatrices))

def SeriesNetworks(networks):
    """
    Series Connection of Networks (reference pins of 1. network is connected to alive pins of 2. network),  input and output  is ABCD matrices of networks
    """
    zmatrices = [ABCD2Z(M) for M in networks]
    return Z2ABCD(reduce(op.add, zmatrices))

def Snormalize(S, Zold, Znew):
    """
    Zold,  Znew port_sayisi uzunlugunda dizilerdir
    """
    ps = len(Zold)
    A = np.matrix(np.zeros((ps, ps)), dtype = complex)
    gamma = np.matrix(np.zeros((ps, ps)), dtype = complex)
    imp_yeni = []
    for i in range(ps):
        imp_yeni.append(Znew[i])
        z = Zold[i]
        ri = ((imp_yeni[i] - z)/(imp_yeni[i] + z))
        gamma[i, i] = ri                
        A[i, i] = csqrt(z.real/imp_yeni[i].real) * abs(imp_yeni[i]/z) * 2 * z/(imp_yeni[i] + z)   
    return A.I * (S - gamma) * (np.matrix(np.eye(ps)) - gamma * S).I * A