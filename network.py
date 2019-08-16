#-*-coding:utf-8-*-
"""
Network Parameters
"""
from builtins import range
import numpy as np
import operator as op
from numpy.lib.scimath import sqrt as csqrt
from functools import reduce

def shunt_z(Z): 
    """
    ABCD parameters of shunt impedance
    """
    return np.matrix([[1., 0], [(1./Z), 1.]])
def series_z(Z): 
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
def pi_network(Zp1, Zs, Zp2): 
    """
    ABCD parameters of Pi network
    """
    return np.matrix([[1 + Zs/Zp2, Zs], [1./Zp1 + 1./Zp2 + Zs/Zp1/Zp2, 1 + Zs/Zp1]])
    
def abcd2y(M): 
    """
    ABCD parameters to Y - Parameters conversion
    """
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    return np.matrix([[d/b, (b * c - a * d)/b], [ -1./b, a/b]])
    
def y2abcd(M): 
    """
    Y-Parameters to ABCD parameters conversion
    """
    y11, y12, y21, y22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    return np.matrix([[ -y22/y21,  -1./y21], [(y12 * y21 - y11 * y22)/y21,  -y11/y21]])

def t2s(M): 
    """
    Transfer scattering parameters to S-Parameters conversion
    According to definition [b1,a1]=T.[a2,b2]
    Ref: https://en.wikipedia.org/wiki/Scattering_parameters#Scattering_transfer_parameters
    """
    t11, t12, t21, t22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    delta=t11*t22-t12*t21
    return np.matrix([[ t12/t22,  delta/t22], [1./t22,  -t21/t22]])

def s2t(M): 
    """
    S-Parameters to Transfer scattering parameters conversion
    According to definition [b1,a1]=T.[a2,b2]
    Ref: https://en.wikipedia.org/wiki/Scattering_parameters#Scattering_transfer_parameters
    """
    s11, s12, s21, s22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    delta=s11*s22-s12*s21
    return np.matrix([[ -delta/s21,  s11/s21], [-s22/s21,  1./s21]])
    
def abcd2z(M): 
    """
    ABCD parameters to Z - Parameters conversion
    """
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    return np.matrix([[a/c, ( - b * c + a * d)/c], [1./c, d/c]])
    
def z2abcd(M): 
    """
    Z - Parameters to ABCD parameters conversion
    """
    z11, z12, z21, z22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    return np.matrix([[z11/z21, (z11 * z22 - z21 * z12)/z21], [1./z21, z22/z21]])
    
def abcd2s(M, Zo=50.0): 
    """
    ABCD parameters to S - Parameters conversion
    Valid for real Zo value
    """
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    s11 = ((a + b/Zo - c * Zo - d)/(a + b/Zo + c * Zo + d))
    s12 = 2. * (a * d - b * c)/(a + b/Zo + c * Zo + d)
    s21 = (2./(a + b/Zo + c * Zo + d))
    s22 = (( - a + b/Zo - c * Zo + d)/(a + b/Zo + c * Zo + d))
    return np.matrix([[s11, s12], [s21, s22]])

def s2abcd(M, Z=[50.0, 50.0]): 
    """
    S-Parameters to ABCD parameters conversion
    Valid for real Z values
    Z: reference impedance list [Z1, Z2]
    """
    Zo1, Zo2 = tuple(Z)
    s11, s12, s21, s22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    a = ((Zo1 + s11 * Zo1) * (1. - s22) + s12 * s21 * Zo1)/(2. * s21 * csqrt(Zo1 * Zo2))
    b = ((Zo1 + s11 * Zo1) * (Zo2 + s22 * Zo2) - s12 * s21 * Zo1 * Zo2)/(2. * s21 * csqrt(Zo1 * Zo2))
    c = ((1. - s11) * (1. - s22) - s12 * s21)/(2. * s21 * csqrt(Zo1 * Zo2))
    d = ((1. - s11) * (Zo2 + s22 * Zo2) + s12 * s21 * Zo2)/(2. * s21 * csqrt(Zo1 * Zo2))
    return np.matrix([[a, b], [c, d]])    

def abcd2t(M, Zo=50.0): 
    """
    ABCD parameters to T - Parameters conversion

    ABCD: [V1 I1]=ABCD*[V2 -I2]
    Pseudo-Wave or Power-Wave? Don't use.
    """
    X = abcd2s(M, Zo)
    return s2t(M)
    
def abcd_change_ports(M): 
    """
    Switching ports of ABCD parameters 
    """
    M = M.I
    np.matrix([[M[0, 0],  - M[0, 1]], [ - M[1, 0], M[1, 1]]])

def t2abcd(M, Z=[50.0,50.0]): 
    """
    T-parameters to ABCD parameters conversion
    """
    X = t2s(M)
    return s2abcd(X, Z)

def cascade_networks(networks): 
    """
    Cascading 2-port Networks,  input and output is ABCD matrices of networks
    """
    return reduce(op.mul, networks)

def parallel_networks(networks):
    """
    Paralleling 2-port Networks,  input and output  is ABCD matrices of networks
    """
    ymatrices = [abcd2y(M) for M in networks]
    return y2abcd(reduce(op.add, ymatrices))

def series_networks(networks):
    """
    Series Connection of Networks (reference pins of 1. network is connected to alive pins of 2. network),  input and output  is ABCD matrices of networks
    """
    zmatrices = [abcd2z(M) for M in networks]
    return z2abcd(reduce(op.add, zmatrices))

def s_normalize_pseudo(S, Zold, Znew):
    """
    Zold,  Znew port_sayisi uzunlugunda dizilerdir
    Pseudo-Wave icin
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
    
def s_normalize_power(S, Zold, Znew):
    """
    Zold,  Znew port_sayisi uzunlugunda dizilerdir
    Power-Wave icin
    Reference: Article, “Multiport conversions between S, Z, Y, h, ABCD, and T parameters”
    """
    ps = len(Zold)
    A = np.matrix(np.zeros((ps, ps)), dtype = complex)
    gamma = np.matrix(np.zeros((ps, ps)), dtype = complex)
    imp_yeni = []
    for i in range(ps):
        imp_yeni.append(Znew[i])
        z = Zold[i]
        gamma[i, i] = ((imp_yeni[i] - z)/(imp_yeni[i] + z.conj()))          
        A[i, i] = 2*csqrt(z.real)*csqrt(imp_yeni[i].real) /(imp_yeni[i].conj() + z)   
    return A.I * (S - gamma.conj()) * (np.matrix(np.eye(ps)) - gamma * S).I * A.conj()
    
def s_phase_deembed(S, phase):
    """
    S-parameter deembedding 
    S is numpy.matrix NxN
    phase, deembedding phase for each port in radian. Positive phase is deembedding into the circuit
    """
    PhaseMatrix=np.exp(1j*np.matrix(np.diag(phase)))   
    return PhaseMatrix*S*PhaseMatrix