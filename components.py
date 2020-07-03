# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:52:33 2009

@author: Tuncay
"""
from constants import *
# from scipy.special import iv, scipy.special.ellipk, scipy.special.ellipe
# from scipy.optimize import scipy.optimize.brentq
# from scipy.integrate import scipy.integrate.quad
# from scipy.misc import scipy.misc.factorial
import scipy.special
import re
import os
# from filters import *
from network import *
# import numpy as np
# from numpy import np.round, np.zeros, np.ones, sin, cos, shape, array, log, log10, exp
# from numpy import fabs, pi, arange, linspace, power, arccosh, flipud, cosh, ceil, sqrt
# from numpy import min as np.min
# from numpy import max as np.max
# from numpy import sum as np.sum
from genel import *
from numpy.lib.scimath import sqrt as csqrt #this works with negative real numbers unlike numpy.sqrt
from transmission_lines import *
# import visvis

co = speed_of_light_in_freespace.simplified.magnitude
eta0 = free_space_wave_impedance.simplified.magnitude
mu0 = free_space_permeability.simplified.magnitude
eps0 = free_space_permittivity.simplified.magnitude

def Zo_eeff_WireOnGroundedSubstrate(arg, defaultunits=[]):
    """ Impedance and Effective Permittivity of Straight Wire Over Substrate.

    Args:
        arg(list): First 4 arguments are inputs.

            1.  Wire Diameter (d);length
            2.  Dielectric Thickness (t);length
            3.  Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>) ;
            4.  Impedance ; impedance
            5.  <font size=+2>&epsilon;<sub>eff</sub></font> ;
            Reference:  Transmission Line Design Handbook, Wadell, s.151
            Note: eeff is the same as eeff of microstrip with w=2*d, t=0

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:3]
    newargs = convert2pq(arg, defaultunits)
    d, h, er = tuple(newargs)
    eeff = ((er + 1.)/ 2.0) + (er - 1.) / 2.0 * \
        ((1. + 12. * h / 2.0 / d) ** (-0.5) + 0.04 * (1. - 2.0 * d / h) ** 2)
    Zo = eta0 / 2 / pi / csqrt(eeff) * np.arccosh(2.0 * h / d)
    arg.append(prettystring(Zo, defaultunits[3]))
    arg.append(prettystring(eeff, defaultunits[4]))
    return arg


def L_StraightRoundWire(arg, defaultunits=[]):
    """ Inductance of a straight round wire.

    Args:
        arg(list): First 5 arguments are inputs.

            1. Wire Diameter ;length
            2. Wire Length ;length
            3. Frequency ; frequency
            4. Dielectric Permeability (<font size=+2>&epsilon;<sub>r</sub></font>)  ;
            5. Conductivity ; electrical conductivity
            6. Inductance ;inductance
            7. Impedance ; impedance
            Reference:  Transmission Line Design Handbook, Wadell, s.380

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """

    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:5]
    newargs = convert2pq(arg, defaultunits)
    d, l, f, mur, sigma = tuple(newargs)
    d = d * 100
    l = l * 100
    x = d * pi * csqrt(2 * mur * mu0 * f / sigma)
    T = csqrt(((0.873011 + 0.00186128 * x)/ (1 - 0.278381 * x + 0.127964 * x * x)))
    L = 0.002 * l * (np.log(4 * l / d) - 1.0 + d / 2.0 / l + mur * T / 4.0)
    X = 2 * pi * f * L * 1e-6
    L = pq.Quantity(L, "uH")
    X = pq.Quantity(X, "ohm")
    arg.append(prettystring(L, defaultunits[5]))
    arg.append(prettystring(X, defaultunits[6]))
    return arg


def Zo_eeff_StraightWireOverSubstrate(arg, defaultunits=[]):
    """ Impedance and Effective Permittivity of Straight Wire Over Substrate.

    Args:
        arg(list): First 4 arguments are inputs.

            1.  Wire Diameter (d);length
            2.  Height Of Wire Center Above Ground (h);length
            3.  Dielectric Thickness (t);length
            4.  Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>) ;
            5.  Impedance ; impedance
            6.  <font size=+2>&epsilon;<sub>eff</sub></font> ;
            Reference:  Transmission Line Design Handbook, Wadell, s.151

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:4]
    newargs = convert2pq(arg, defaultunits)
    d, h, b, er = tuple(newargs)
    eeff = (np.log(4 * h / d)/ (np.log(4 * (h - b) / d + 4 * b / d / er)))
    R = (2.0/ (8 * h / d - d / h / 2.0))
    u = (1.0/ ((4 * h / d) * (4 * h / d) - 1.0))
    Zo = eta0 / 2 / pi / csqrt(eeff) * np.arccosh((1 - u * u) / 2.0 / R + (R/ 2.0))
    arg.append(prettystring(Zo, defaultunits[4]))
    arg.append(prettystring(eeff, defaultunits[5]))
    return arg


def L_StraightFlatWire(arg, defaultunits=[]):
    """ Inductance of a flat wire.

    Args:
        arg(list): First 6 arguments are inputs.

            1.  Wire Width ;length
            2.  Wire Thickness ;length
            3.  Wire Length ;length
            4.  Frequency ; frequency
            5.  Relative Permeability ;
            6.  Conductivity ; electrical conductivity
            7.  Inductance ;inductance
            8.  Impedance ;impedance
            Reference:  Transmission Line Design Handbook, Wadell, s.382

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:6]
    newargs = convert2pq(arg, defaultunits)
    w, t, l, f, mur, sigma = tuple(newargs)
    x = t * pi * csqrt(2 * mur * mu0 * f / sigma)  # ???
    T = csqrt(
        ((0.873011 + 0.00186128 * x)/ (1.0 - 0.278381 * x + 0.127964 * x * x)))
    L = 0.002 * l * \
        (np.log(2.0 * l / (w + t)) + 0.25049 + (w + t) / 3.0 / l + mur * T / 4.0)
    X = 2.0 * pi * f * L * 1.0e-9
    L = pq.Quantity(L, "nH")
    X = pq.Quantity(X, "ohm")
    arg.append(prettystring(L, defaultunits[6]))
    arg.append(prettystring(X, defaultunits[7]))
    return arg


def L_microstrip_via_hole(arg, defaultunits=[]):
    """ Inductance of a via hole in microstrip.

    Args:
        arg(list): First 2 arguments are inputs.

            1. Via Radius ;length
            2. Substrate Thickness ;length
            3. Inductance ; inductance
            Reference:  Microstrip Via Hole Grounds in Microstrip.pdf

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:2]
    newargs = convert2pq(arg, defaultunits)
    r, h = tuple(newargs)
    ind = mu0 / 2 / pi * \
        (h * np.log(((h + csqrt(h * h + r * r))/ r)) +
         1.5 * (r - csqrt(h * h + r * r)))
    arg.append(prettystring(ind, defaultunits[2]))
    return arg


def L_air_core_coil(arg, defaultunits=[]):
    """ Inductance of a via hole in microstrip.

    Args:
        arg(list): First 4 arguments are inputs.

            1. Wire Diameter (d) ;length
            2. Coil Inner Diameter (d_in) ;length
            3. Spacing Between Turns (s) ; length
            4. Number Of Turns ;
            5. Inductance ; inductance
            6. Resonance Frequency ; frequency
            Reference:  www.microwavecoil.com , Microwave Components Inc.

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:4]
    newargs = convert2pq(arg, defaultunits)
    d, d_in, s, N = tuple(newargs)
    d = d * coef("inch")
    d_in = d_in * coef("inch")
    s = s * coef("inch")
    ind = 17 * N ** 1.3 * (d_in + d) ** 1.7 / (d + s) ** 0.7 * 1e-9
    z = N * d + (N - 1) * s
    L = (csqrt(z ** 2 + (N * pi * (d_in + d)) ** 2)/ coef("inch"))
    # For resonance, coil length should be half of the wavelength
    SRF = 0.5 * co / L
    argout = [ind, SRF]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg


def L_BondWire(arg, defaultunits=[]):
    """ Inductance of a bond wire.

    Args:
        arg(list): First 4 arguments are inputs.

            1. Bondwire Radius ;length
            2. Substrate Thickness ;length
            3. Distance Between End Points ;length
            4. Angle At End Points In Degrees ; angle
            5. Inductance ;inductance
            Reference:  Transmission Line Design Handbook, Wadell, s.153

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:4]
    newargs = convert2pq(arg, defaultunits)
    r, h, d, beta = tuple(newargs)
    d = d * coef("inch")
    h = h * coef("inch")
    r = r * coef("inch")
    beta = beta * coef("rad")
    L = 10.0 * scipy.integrate.quad(lambda x: np.log(2.0 / r * (csqrt((d / 2.0 / np.sin(beta))
                    ** 2. - x ** 2.) + (h - d / 2.0 / np.tan(beta)))), 0, (d/ 2.0))[0]
    L = pq.Quantity(L, "nH")
    arg.append(prettystring(L, defaultunits[4]))
    return arg


def Chebyshev_QWave_Impedance_Transformer(arg, defaultunits=[]):
    """ Chebyshev Quarter Wave Impedance Transformer.

    Args:
        arg(list): First 6 arguments are inputs.

            1.  Source Impedance ; impedance
            2.  Load Impedance ; impedance
            3.  Number Of Matching Sections ;
            4.  Minimum Frequency ; frequency
            5.  Maximum Frequency ; frequency
            6.  Test Frequency ; frequency
            7.  Impedances ; impedance
            8.  Return Loss at Test Frequency ;
            Reference:  Impedance Matching and Transformation.pdf + eski kod

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:6]
    newargs = convert2pq(arg, defaultunits)
    Z0, ZL, N, fmin, fmax, ftest = tuple(newargs)
    BW = (fmax/ fmin)
    r = (ZL/ Z0)
    N = int(np.round(N))
    print("N ", N)
    N = N + 1
    L = np.zeros([N + 1, N + 2])
    a = np.ones([N])
    m = np.ones([N])
#    z=np.ones([N+1])
    z = [1]
    fi = (pi/ (BW + 1.))
    L[1, 1] = (1.0/ np.cos(fi))
    L[0, 0] = 2.0 * np.ones(np.shape(L[1, 1]))
    for i in range(3, N + 1):
        L[i - 1, 0] = 2 * L[i - 2, 1] * L[1, 1] - L[i - 3, 0]
        for j in range(2, N + 1):
            L[i - 1, j - 1] = (L[i - 2, j - 2] + L[i - 2, j]) * \
                L[1, 1] - L[i - 3, j - 1]
    if (N % 2 == 0):
        for i in range(1, N//2 + 1):
            a[i - 1] = L[N - 1, N + 1 - 2 * i]
            a[N - i] = a[i - 1]
    else:
        for i in range(1, (N + 1)//2 + 1):
            a[i - 1] = L[N - 1, N + 1 - 2 * i]
            a[N - i] = a[i - 1]
    a = a/np.sum(a, 0)
    print("a= ", a)
#    m[:]=a[:]*np.log(r)/np.sum(a)
    m = np.array([a]).T * np.log(r)
#    m=a*np.log(r)
#    print Z0,np.ones(np.shape(m)),Z0*np.ones(np.shape(m))
    z[0] = Z0
#    z[0]=Z0*np.ones(np.shape(m))
    for i in range(1, N + 1):
        z.append(z[-1] * np.exp(m[i - 1]))
    z = np.array(z)

    thetam = (2. - 2. * (BW - 1.) / (BW + 1.)) * pi / 4.
    print("s ", r)
    print("s ", N)
    print("s ", thetam)
#    max_gamma=abs((r-1)/(r+1)/chebyt(N-1)(1./np.cos(thetam)))
    gamma = abs((r - 1) / (r + 1) * chebyt(N - 1)
                ((np.cos(pi / (fmin + fmax) * ftest)/ np.cos(thetam))) / chebyt(N - 1)((1./ np.cos(thetam))))
    arg.append(globsep2.join([prettystring(x, defaultunits[6]) for x in z]))
    arg.append(prettystring(20. * np.log10(gamma), defaultunits[7]))
    return arg


def Binomial_QWave_Impedance_Transformer(arg, defaultunits=[]):
    """ Binomial Quarter Wave Impedance Transformer.

    Args:
        arg(list): First 5 arguments are inputs.

            1.  Source Impedance;impedance
            2.  Load Impedance;impedance
            3.  Number Of Matching Sections;
            4.  Max(dB(S<sub>11</sub>)) In Frequency Band ;
            5.  Center Frequency ; frequency
            6.  Impedances ; impedance
            7.  Bandwidth ; frequency
            Reference:  Impedance Matching and Transformation.pdf

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:5]
    newargs = convert2pq(arg, defaultunits)
    Z0, ZL, N, gamma, fcenter = tuple(newargs)
    r = (ZL/ Z0)
    N = int(np.round(N))
    gamma = 10. ** ((gamma/ 20.))
    k = np.log(r)
    z = [Z0]
    a = 0
    for i in range(1, N + 1):
        a = 2. ** (-N) * k * scipy.misc.factorial(N) / \
            scipy.misc.factorial(N - i + 1) / scipy.misc.factorial(i - 1)
        # z.append(z[i-1]*(1+a)/(1-a))
        z.append(z[i - 1] * np.exp(a))
    z.append(ZL)
    BW = (2. - 4. / pi * arcnp.cos(0.5 *
          ((gamma/ abs(2. ** (-N) * (r - 1) / (r + 1)))) ** ((1./ N)))) * fcenter
    arg.append(globsep2.join([prettystring(x, defaultunits[4]) for x in z]))
    arg.append(prettystring(BW, defaultunits[6]))
    return arg


def Tee_Attenuator_Synthesis(arg, defaultunits=[]):
    """ Tee Attenuator Synthesis.

    Args:
        arg(list): First 5 arguments are inputs.

            1. Reference Impedance (Zo); impedance
            2. Series Impedance (Rs); impedance
            3. Parallel Impedance (Rp); impedance
            4. <font size=+1>S<sub>11</sub></font> ;
            5. <font size=+1>S<sub>21</sub></font> ;
            6. P1 ;
            7. P2 ;
            8. P3 ;
            Reference:

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:5]
    newargs = convert2pq(arg, defaultunits)
    Zo, dummy, dummy, dummy, S21 = tuple(newargs)
    K = 10.0 ** ((S21/ 20))
    R1 = Zo * (1.0 - K) / (1.0 + K)  # Series resistor
    R2 = 2 * K * Zo / (1.0 - K * K)  # Parallel resistor
    arg[2] = (prettystring(R2, defaultunits[2]))
    arg[1] = (prettystring(R1, defaultunits[1]))
    return arg


def Tee_Attenuator_Analysis(arg, defaultunits=[]):
    """ Tee Attenuator Analysis.

    Args:
        arg(list): First 3 arguments are inputs.

            1. Reference Impedance (Zo); impedance
            2. Series Impedance (Rs); impedance
            3. Parallel Impedance (Rp); impedance
            4. <font size=+1>S<sub>11</sub></font> ;
            5. <font size=+1>S<sub>21</sub></font> ;
            6. P1 ;
            7. P2 ;
            8. P3 ;
            Reference:

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 3
    arg = arg[:3]
    newargs = convert2pq(arg, defaultunits)
    Zo, Rs, Rp = tuple(newargs)
    Zx = (Rs + Zo) * Rp / (Rs + Zo + Rp)
    Z1 = Zx + Rs
    S11 = 20. * np.log10((np.fabs(Z1 - Zo)/ (Z1 + Zo)))
    K = (Rs + Zo) * Rp / (Rs + Zo + Rp) / Z1 * Zo / (Zo + Rs)
    Gamma = ((Z1 - Zo)/ (Z1 + Zo))
    S21 = 20. * np.log10(K * (1.0 + Gamma))
    P1 = 20 * np.log10((Rs/ Z1))
    P2 = 10 * np.log10(((Zx/ Z1)) ** 2 * Zx / Rp)
    P3 = 10 * np.log10(((Zx/ Z1)) ** 2 * (1 - (Zx/ Rp)) * Rs / (Rs + Zo))
    argout = [S11, S21, P1, P2, P3]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg


def Pi_Attenuator_Synthesis(arg, defaultunits=[]):
    """ Pi Attenuator Analysis.

    Args:
        arg(list): First 3 arguments are inputs.

            1. Reference Impedance (Zo); impedance
            2. Series Impedance (Rs); impedance
            3. Parallel Impedance (Rp); impedance
            4. <font size=+1>S<sub>11</sub></font> ;
            5. <font size=+1>S<sub>21</sub></font> ;
            6. P1 ;
            7. P2 ;
            8. P3 ;
            Reference:

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:5]
    newargs = convert2pq(arg, defaultunits)
    Zo, dummy, dummy, dummy, S21 = tuple(newargs)
    print(Zo)
    K = 10.0 ** ((S21/ 20.0))
    R1 = Zo * (1 - K * K) / 2. / K  # Series resistor
    R2 = Zo * (1 + K) / (1 - K)     # Parallel resistor
    arg[1] = prettystring(R1, defaultunits[1])
    arg[2] = prettystring(R2, defaultunits[2])
    return arg


def Pi_Attenuator_Analysis(arg, defaultunits=[]):
    """ Pi Attenuator Analysis.

    Args:
        arg(list): First 3 arguments are inputs.

            1. Reference Impedance (Zo); impedance
            2. Series Impedance (Rs); impedance
            3. Parallel Impedance (Rp); impedance
            4. <font size=+1>S<sub>11</sub></font> ;
            5. <font size=+1>S<sub>21</sub></font> ;
            6. P1 ;
            7. P2 ;
            8. P3 ;
            Reference:

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 3
    arg = arg[:3]
    newargs = convert2pq(arg, defaultunits)
    Zo, Rs, Rp = tuple(newargs)
    Zt = Rp * Zo / (Rp + Zo) + Rs
    Z1 = Zt * Rp / (Zt + Rp)
    S11 = 20. * np.log10((abs(Z1 - Zo)/ (Z1 + Zo)))
    K = Rp * Zo / (Rp + Zo) / Zt
    Gamma = ((Z1 - Zo)/ (Z1 + Zo))
    S21 = 20. * np.log10(K * (1 + Gamma))
    # power dissipated at first resistor
    P1 = 10. * np.log10((1 + Gamma) ** 2 * Zo / Rp)
    # power dissipated at last resistor
    P3 = 10. * np.log10((K * (1 + Gamma)) ** 2 * Zo / Rp)
    # power dissipated at middle resistor
    P2 = 10. * np.log10(((1 + Gamma) * (1 - K ** 2)) ** 2 * Zo / Rs)
    argout = [S11, S21, P1, P2, P3]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg


def Bridged_Tee_Attenuator_Synthesis(arg, defaultunits=[]):
    """ Bridged Tee Attenuator Synthesis.

    Args:
        arg(list): First 3 arguments are inputs.

            1. Reference Impedance (Zo); impedance
            2. Series Impedance (Rs); impedance
            3. Parallel Impedance (Rp); impedance
            4. <font size=+1>S<sub>11</sub></font> ;
            5. <font size=+1>S<sub>21</sub></font> ;
            Reference:

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:5]
    newargs = convert2pq(arg, defaultunits)
    Zo, dummy, dummy, dummy, S21 = tuple(newargs)
    K = 10.0 ** ((S21/ 20.0))
    R1 = Zo * (1 - K) / K  # Series resistor
    R2 = Zo * Zo / R1      # Parallel resistor
    arg[1] = prettystring(R1, defaultunits[1])
    arg[2] = prettystring(R2, defaultunits[2])
    return arg


def Bridged_Tee_Attenuator_Analysis(arg, defaultunits=[]):
    """ Bridged Tee Attenuator Analysis.

    Args:
        arg(list): First 3 arguments are inputs.

            1. Reference Impedance (Zo); impedance
            2. Series Impedance (Rs); impedance
            3. Parallel Impedance (Rp); impedance
            4. <font size=+1>S<sub>11</sub></font> ;
            5. <font size=+1>S<sub>21</sub></font> ;
            Reference:

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:3]
    newargs = convert2pq(arg, defaultunits)
    Zo, Rs, Rp = tuple(newargs)
    s = Triangle2StarTransformation(
        [prettystring(x) for x in [0,0,0,Rs, Zo, Zo]], ["ohm", "ohm", "ohm", "ohm", "ohm", "ohm"])
    s = convert2pq(s)
    R1 = s[0]
    R2 = s[2] + Rp
    temp = Tee_Attenuator_Analysis(
        [prettystring(x) for x in [Zo, R1, R2]], ["ohm", "ohm", "ohm", "", "", "", "", ""])
    s11 = temp[3]
    s21 = temp[4]
    argout = [s11, s21]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg


def DualFrequencyTransformer(arg, defaultunits=[]):
    """ Dual Frequency Transformer.

    Args:
        arg(list): First 4 arguments are inputs.

            1. Source Impedance; impedance
            2. Load Impedance; impedance
            3. f1 Lower Frequency; frequency
            4. f2 Higher Frequency; frequency
            5. Z1; impedance
            6. Z2; impedance
            7. Electrical Length ; angle
            Reference:  A Small Dual Frequency Transformer in Two Sections

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:4]
    newargs = convert2pq(arg, defaultunits)
    Zo, ZL, f1, f2 = tuple(newargs)
    a = np.tan(pi * f1 / (f1 + f2)) ** 2.0
    b = 0.5 * Zo * (ZL - Zo) / a
    Z1 = csqrt(b + csqrt(b ** 2.0 + Zo ** 3.0 * ZL))
    Z2 = Zo * ZL / Z1
    theta = np.pi * f1 / (f1 + f2)  # electrical length at f1
    argout = [Z1, Z2, theta]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def SymmetricLangeCoupler(arg, defaultunits=[]):
    """ Symmetric Lange Coupler.

    Args:
        arg(list): First 3 arguments are inputs.

            1. C: Voltage coupling coefficient in dB (positive);
            2. n: Number of fingers (should be even);
            3. Reference Impedance;impedance
            4. Zoo;impedance
            5. Zoe;impedance
            Reference:  Microwave Circuits, Analysis and Computer-Aided Design, Fusco

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:3]
    newargs = convert2pq(arg, defaultunits)
    C, n, Zo = tuple(newargs)
    n = np.round(n)
    C = 10 ** ((-C/ 20.0))
    if divmod(n, 2)[1] != 0:
        print("n should be an even number!")
    q = csqrt(C * C + (1. - C * C) * (n - 1) ** 2)
    a = csqrt(((1. - C)/ (1. + C)))
    b = (n - 1) * (1. + q) / ((C + q) + (n - 1) * (1. - C))
    Zoo = Zo * a * b
    Zoe = Zoo * (q + C) / (n - 1) / (1. - C)
    arg.append(prettystring(Zoo, defaultunits[3]))
    arg.append(prettystring(Zoe, defaultunits[4]))
    return arg


def AWG2Dia(arg, defaultunits=[]):
    """ Convert AWG to Diameter.

    Args:
        arg(list): First 1 arguments are inputs.

            1.  AWG ;
            2. Diameter ;length
            3. Current rating in still air ; current
            Reference:  Wikipedia, Current rating is calculated through curve fit from online data

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:1]
    newargs = convert2pq(arg, defaultunits)
    x = tuple(newargs)[0]
    #a = np.exp(2.1104 - 0.11594 * x) * 1e-3 * pq.m
    a = np.exp(2.1104 - 0.11594 * x) * 1e-3
    x = a * 1000.0
    I = 0.00116740089661*x**8 -0.0408734276969*x**7 + 0.5878919812*x**6-4.46963440886*x**5 + 19.2489764699*x**4 \
            -46.8120244881*x**3 + 62.0053391882*x**2 -16.3199167738*x**1 + 1.37792724313
    arg.append(prettystring(a, defaultunits[1]))
    arg.append(prettystring(I, defaultunits[2]))
    return arg


def Dia2AWG(arg, defaultunits=[]):
    """ Convert Diameter to AWG.

    Args:
        arg(list): First 1 arguments are inputs.

            1. AWG ;
            2. Diameter ;length
            3. Current rating in still air ; current
            Reference:  Wikipedia

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:2]
    newargs = convert2pq(arg, defaultunits)
    n, x = tuple(newargs)
    n = pq.Quantity(((2.1104 - np.log(1000 * x))/ 0.11594), "")
    x = x * 1000.0
    I = 0.00116740089661*x**8 -0.0408734276969*x**7 + 0.5878919812*x**6-4.46963440886*x**5 + 19.2489764699*x**4 \
        -46.8120244881*x**3 + 62.0053391882*x**2 -16.3199167738*x**1 + 1.37792724313
    arg[0] = prettystring(n, defaultunits[0])
    arg.append(prettystring(I, defaultunits[2]))
    return arg


def PCBTrackCurrentCapacityIPC(arg, defaultunits=[]):
    """ PCB Track Current Capacity, IPC.

    Args:
        arg(list): First 4 arguments are inputs.

            1. Metal Width;length
            2. Metal Thickness;length
            3. Allowable Temperature Rise; temperature
            4. External if 1, Internal if 0;
            5. Current ; current
            Reference:  IPC2221A

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:4]
    newargs = convert2pq(arg, defaultunits)
    w, t, dT, external = tuple(newargs)
    temp = (w * 1000) * 1000.0 / 25.4 * (t * 1000) * 1000.0 / 25.4
    current = 0.024 * pow(dT, 0.44) * pow(temp, 0.725) * (external + 1)
    current = current * pq.A
    arg.append(prettystring(current, defaultunits[4]))
    return arg


def PCBTrackCurrentCapacity(arg, defaultunits=[]):
    """ PCB Track Current Capacity.

    Args:
        arg(list): First 7 arguments are inputs.

            1. Metal Width;  length
            2. PCB Height;     length
            3. Metal Thickness;        length
            4. Allowable Temperature Rise; temperature
            5. Thermal Conductivity;  thermal conductivity
            6. Electrical Conductivity; electrical conductivity
            7. External if 1, Internal if 0;
            8. Current ; current
            Reference:

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:7]
    newargs = convert2pq(arg, defaultunits)
    w, h, t, dT, K, sigma, external = tuple(newargs)
    external = round(external)
    A = dT * w * w * K * sigma * t / h
    cur = csqrt(A * (external + 1.0) / 2.0) * pq.A
    arg.append(prettystring(cur, defaultunits[7]))
    return arg


def OptimumMitered90DegMicrostripBend(arg, defaultunits=[]):
    """ Optimum Mitered Microstrip Bend Parameters.

    Args:
        arg(list): First 2 arguments are inputs.

            1.  Microstrip Width;length
            2.  Substrate Height;length
            3.  Miter Length; length
            Reference: Tranmission line design handbook, p.290

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:2]
    newargs = convert2pq(arg, defaultunits)
    w, h = tuple(newargs)
    print("wh", w, h)
    x = w * csqrt(2.0) * (1.04 + 1.3 * np.exp(-1.35 * w / h))
    print(x)
    arg.append(prettystring(x, defaultunits[2]))
#    arg.append(prettystring((2.1104-np.log(1000*x))/0.11594,""))
    return arg


def OptimumMiteredArbitraryAngleMicrostripBend(arg, defaultunits=[]):
    r""" Optimum Mitered Microstrip Bend Parameters.

    Args:
        arg(list): First 2 arguments are inputs.

            1.  Microstrip Width;length;
            2.  Substrate Height;length;
            3.  Angle (0-180 degrees); angle ;
            4.  Miter Length; length ;
            Reference: MWOHELP, MBENDA model


    Burada scipy.interpolate.griddata kullanildi ve maalesef extrapolation yapmiyor. Sinir disi degerlerde dogrudan en yakin deger kullanildi.

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:3]
    newargs = convert2pq(arg, defaultunits)
    w, h, angle = tuple(newargs)
    angle = angle / pi * 180.0
    a = [0.5, 1.0, 2.0]
    b = [0, 30, 60, 90, 120]
    dd = np.meshgrid(a, b)
    z = np.array(
        [[0, 0, 0], [12, 19, 7], [45, 41, 31], [75, 63, 56], [98, 92, 79]])
    # from matplotlib.mlab import griddata
    # this library gives NaN value for values out of bounds
    from scipy.interpolate import griddata
    wh = (w/ h)
    print(wh)
    print(angle)
    # x=griddata(dd[0].flatten(),dd[1].flatten(),z.flatten(),np.array([w/h]),np.array([angle]))[0][0]
    # # Matplotlib fonksiyonu icin
    if (wh <= 2.0 and wh >= 0.0 and angle <= 120.0 and angle >= 0.0):
        x = griddata(list(zip(dd[0].flatten(), dd[1].flatten())),
                     z.flatten(), (wh, angle), method='cubic')
    else:
        x = griddata(list(zip(dd[0].flatten(), dd[1].flatten())),
                     z.flatten(), (wh, angle), method='nearest')
    print(x)
    arg.append(prettystring(csqrt(2.0) * w * x / 100.0, defaultunits[3]))
    return arg


def Interference_Phase_Amp_Error(arg, defaultunits=[]):
    r""" Maximum phase and amplitude variation of a signal in presence of an interfering signal.

    Args:
        arg(list): First 1 arguments are inputs.

            1. Difference in dB ;
            2. Amplitude Error;
            3. Phase Error; angle
            Reference:

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:1]
    newargs = convert2pq(arg, defaultunits)
    farkdB = tuple(newargs)
    x = 10.0 ** ((-abs(farkdB)/ 20.0))
    arg.append(prettystring(20 * np.log10(((1. + x)/ (1.- x))), ""))
    arg.append(prettystring(2.0 * arcnp.sin(x) / pi * 180.0, ""))
    return arg


def ParallelPlateCap(arg, defaultunits=[]):
    """ Parallel Plate Capacitance.

    Args:
        arg(list): First 4 arguments are inputs.

            1. Width;length
            2. Length;length
            3. Height;length
            4. Dielectric Permittivity;
            5. Capacitance; capacitance
            Reference:

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:4]
    newargs = convert2pq(arg, defaultunits)
    w, l, h, er = tuple(newargs)
    cap = er * eps0 * w * l / h
    arg.append(prettystring(cap, defaultunits[4]))
    return arg


def CircularPlateCap(arg, defaultunits=[]):
    """ Circular Plate Capacitance.

    Args:
        arg(list): First 3 arguments are inputs.

            1. Radius;length
            2. Height;length
            3. Dielectric Permittivity;
            4. Capacitance; capacitance
            Reference:

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:3]
    newargs = convert2pq(arg, defaultunits)
    r, h, er = tuple(newargs)
    cap = er * eps0 * pi * r ** 2 / h
    arg.append(prettystring(cap, defaultunits[3]))
    return arg


def Shorten90DegreeLine(arg, defaultunits=[]):
    """ Shortening 90 Degree Line with a capacitive load.

    Args:
        arg(list): First 3 arguments are inputs.

            1. Impedance (Z<sub>o</sub>); impedance
            2. Center Frequency ;  frequency
            3. Electrical Length (<font size=+1>&theta;</font>) ; angle
            4. Impedance (Z<sub>x</sub>); impedance
            5. Capacitance ; capacitance
            Reference:

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:3]
    newargs = convert2pq(arg, defaultunits)
    Zo, fcenter, elec = tuple(newargs)
    imp = (Zo/ np.sin(elec))
    cap = ((np.cos(elec)/ (Zo * 2.0 * pi * fcenter)))
    arg.append(prettystring(imp, defaultunits[3]))
    arg.append(prettystring(cap, defaultunits[4]))
    return arg

def Z_WG_TE10(er, a, b, freq, formulation=1):
    kc = (pi/ a)
    k = 2 * pi * freq * csqrt(er) / co
    beta = csqrt(k * k - kc * kc)
    if formulation==1:  # Power-Voltage
        imp = csqrt(mu0 / eps0 / er) * k / beta * 2. * b / a
    elif formulation==2:  # Voltage-Current
        imp = csqrt(mu0 / eps0 / er) * k / beta * pi * b / 2. / a
    elif formulation==3:  # Power-Current
        imp = csqrt(mu0 / eps0 / er) * k / beta * pi**2 * b / 8. / a
    elif formulation==4:  # Wave Impedance, E/H
        imp = csqrt(mu0 / eps0 / er) * k / beta
    return imp


def HomogeneousRectWaveguideParameters_TE(arg, defaultunits=[]):
    """ Homogeneous Rectangular Waveguide Parameters.

    Args:
        arg(list): First 10 arguments are inputs.

            1. Dielectric Permittivity in Waveguide;
            2. Waveguide Width;length
            3. Waveguide Height;length
            4. Mode (0: Te, 1: Tm);
            5. M;
            6. N;
            7. Tand Of Dielectric;
            8. Electrical Conductivity Of Walls; electrical conductivity
            9. Frequency; frequency
            10. Physical Length;length
            11. Cond Loss; loss per length
            12. Diel Loss; loss per length
            13. Cutoff Freq; frequency
            14. Lambda_Guided;length
            15. Impedance; impedance
            16. Electrical Length; angle

            Reference:  Marcuvitz Waveguide Handbook s.253

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    # if len(defaultunits)==0:
        # defaultunits=[""]*len(arg)*2
    arg = arg[:10]
    newargs = convert2pq(arg, defaultunits)
    er, a, b, mode, m, n, tand, sigma, freq, phy_length = tuple(newargs)
    mode = np.round(mode)
    # mode=0 means TE mode, mode=1 means TM mode.
    skin_depth = csqrt((1./ (pi * freq * mu0 * sigma)))
    kc = pi * csqrt((m/ a) ** 2 + (n/ b) ** 2)
    k = 2 * pi * freq * csqrt(er) / co
    beta = csqrt(k * k - kc * kc)
    eta = (eta0/ csqrt(er))
    imp_te = Z_WG_TE10(er, a, b, freq)
    Rs = 1.0 / sigma / skin_depth

    # if (n==0):
        # Pdiss=    (1/sigma/skin_depth)*((a/2+b)+0.5*(k*k-kc*kc)*pi*pi*(m*m/a)/(kc**4))
        # Pt=   0.25*(k*k-kc*kc)*a*b*imp_te*pi*pi*m*m/a/a/(kc**4)
    # elif (m==0):
        # Pdiss=(1/sigma/skin_depth)*((a+b/2)+0.5*(k*k-kc*kc)*pi*pi*(n*n/b)/(kc**4))
        # Pt=   0.25*(k*k-kc*kc)*a*b*imp_te*pi*pi*n*n/b/b/(kc**4)
    # else:
        # Pdiss=(1/sigma/skin_depth)*((a+b)/2+0.5*(k*k-kc*kc)*pi*pi*(n*n/b+m*m/a)/(kc**4))
        # Pt=   0.125*(k*k-kc*kc)*a*b*imp_te*pi*pi*((m/a)**2+(n/b)**2)/(kc**4)

    # alpha_c=0.5*Pdiss/Pt
    # alpha_d=4*pi*pi*freq*freq*mu0*er*eps0*tand/2./beta

    if mode == 0:  # TE mode
        if (n == 0):
            alpha_c = 2 * Rs * \
                (0.5 + b / a * ((kc/ k)) ** 2) / \
                b / eta / csqrt(1. - ((kc/ k)) ** 2)
        else:
            alpha_c = (0.5 + b/a) * ((kc/ k)) ** 2 + (b ** 2 * m ** 2 + a * b *
                                                       n ** 2) / (b ** 2 * m ** 2 + a ** 2 * n ** 2) * (1. - (kc/k) ** 2)
            alpha_c = 2 * Rs * alpha_c / b / eta / csqrt(1. - ((kc/ k)) ** 2)
    elif mode == 1:  # TM mode
        alpha_c = ((b ** 3 * m ** 2 + a ** 3 * n ** 2)/(a * b ** 2 * m ** 2 + a ** 3 * n ** 2))
        alpha_c = 2 * Rs * alpha_c / b / eta / csqrt(1. - ((kc/ k)) ** 2)
    alpha_d = 0.5 * k * tand / csqrt(1. - ((kc/ k)) ** 2)

    cutoff = co / 2.0 / csqrt(er) * csqrt((m/a) ** 2.0 + (n/b) ** 2.0)
    loss_cond = 20.0 * alpha_c * np.log10(np.exp(1))  # dB/m
    loss_diel = 20.0 * alpha_d * np.log10(np.exp(1))  # dB/m
    argout = [loss_cond, loss_diel, cutoff,
              2 * pi / beta, imp_te, phy_length * beta]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg


def InductivePostInWaveguide(arg, defaultunits=[]):
    """ Inductive Post In Waveguide.

    Args:
        arg(list): First 6 arguments are inputs.

            1. Dielectric Permittivity in Waveguide (<font size=+2>&epsilon;<sub>r</sub></font>);
            2. Waveguide Width (a);length
            3. Waveguide Height (b);length
            4. Post Diameter (d);length
            5. Waveguide Sidewall To Post Center (s);length
            6. Frequency; frequency
            7. Inductance;inductance
            8. Capacitance; capacitance
            9. Impedance; impedance
            Reference:  Marcuvitz Waveguide Handbook s.257

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:6]
    newargs = convert2pq(arg, defaultunits)
    er, a, b, d, x, freq = tuple(newargs)
    print("Valid if frequency is between "+str(co/np.sqrt(er)/2.0/a)+" and "+str(co/np.sqrt(er)/a))
    kc = (pi/ a)
    k = 2 * pi * freq * csqrt(er) / co
    Zo = csqrt(mu0 / eps0 / er) * k / np.sqrt(k * k - kc * kc) * \
        2. * b / a  # impedance of waveguide
    lg = 2 * pi / np.sqrt(k ** 2.0 - kc ** 2.0)  # lambda guided
    # if (x/a-0.5)<0.01:
        # temp=sum([(sqrt(n**2-(k*a/pi)**2)-n+2.0/n*(k*a/2.0/pi)**2) for n in range(3,100,2)])
        # S2=np.log(4*a/pi/d)-2.5+11.0/3.0*(pi/k/a)**2-(2*pi/k/a)**2*temp
        # temp=sum([(1.0/sqrt(n**2-(k*a/pi)**2)-1.0/n) for n in range(3,100,2)])
        # So=np.log(4*a/pi/d)-2.0+2*temp
        # Xb=Zo*a/lg*1.0/((a/pi/d)**2.0+11.0/24.0)
        # Xa=Xb/2.0+Zo*a/2.0/lg*(So-(k*d/4.0)**2-5.0/8.0*(k*d/4.0)**4-2.0*(k*d/4.0)**4*(S2-2*So*(1.0-kc**2/k**2))**2)
        # return Xa/Zo*lg/2/a
        # return (Xa/2.0/pi/freq,1.0/2.0/pi/freq/Xb) #L,C
    Xb = Zo * a / lg * (pi * d / a * np.sin(pi * x / a)) ** 2
    if isinstance(x, float) and isinstance(a, float):
        temp = sum([np.sin(n * pi * x / a) ** 2 * ((1.0/ csqrt(n ** 2 - (k * a / pi) ** 2)) - (1.0/ n))
                   for n in range(2, 100)])
    elif not isinstance(x, float):
        temp = np.array([])
        for cc in x:
            temp = np.append(temp, sum(
                [np.sin(n * pi * cc / a) ** 2 * ((1.0/ csqrt(n ** 2 - (k * a / pi) ** 2)) - (1.0/ n)) for n in range(2, 100)]))
    elif not isinstance(a, float):
        temp = np.array([])
        for cc in a:
            temp = np.append(temp, sum(
                [np.sin(n * pi * x / cc) ** 2 * ((1.0/ csqrt(n ** 2 - (k * cc / pi) ** 2)) - (1.0/ n)) for n in range(2, 100)]))
    So = np.log(4 * a / pi / d * np.sin(pi * x / a)) - \
        2.0 * np.sin(pi * x / a) ** 2 + 2.0 * temp
    if isinstance(x, float) and isinstance(a, float):
        temp = sum([np.sin(2 * n * pi * x / a) * (n/np.sqrt(n ** 2 - (k * a / pi) ** 2) - 1.0)
                   for n in range(2, 100)])
    elif not isinstance(x, float):
        temp = np.array([])
        for cc in x:
            temp = np.append(temp, sum(
                [np.sin(2 * n * pi * cc / a) * (n/csqrt(n ** 2 - (k * a / pi) ** 2) - 1.0) for n in range(2, 100)]))
    elif not isinstance(a, float):
        temp = np.array([])
        for cc in a:
            temp = np.append(temp, sum(
                [np.sin(2 * n * pi * x / cc) * (n/csqrt(n ** 2 - (k * cc / pi) ** 2) - 1.0) for n in range(2, 100)]))
    S1 = (0.5/ np.tan(pi * x / a)) - np.sin(2 * pi * x / a) + temp
    deltaX = Zo * a / 2.0 / lg / np.sin(pi * x / a) ** 2 * (So - (k * d / 4.0) **
                                                         2 - (pi * d / 2. / a) ** 2 * ((So/ np.tan(pi * x / a)) - S1) ** 2)  # Xa-Xb/2.0
    Xa = (Xb/ 2.0) + deltaX
    L = Xa / 2.0 / pi / freq / Zo
    C = 1.0 / 2.0 / pi / freq / Xb / Zo
    arg.append(prettystring(L, defaultunits[6]))
    arg.append(prettystring(C, defaultunits[7]))
    arg.append(prettystring(Zo, defaultunits[8]))
    return arg


def InductiveWindowInWaveguide(arg, defaultunits=[]):
    """ Waveguide Width Step from Rectangular Waveguide to Evanescent Mode Rectangular Waveguide.

    Args:
        arg(list): First 6 arguments are inputs.

            1. Dielectric Permittivity in Waveguide (<font size=+2>&epsilon;<sub>r</sub></font>);
            2. Waveguide Width (a);length
            3. Waveguide Height (b);length
            4. Difference Of Waveguide Width To Window Width;length
            5. Window Thickness;length
            6. Frequency; frequency
            7. Inductance;inductance
            8. Capacitance; capacitance
            9. Impedance; impedance

            Reference:  Marcuvitz Waveguide Handbook s.253

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:6]
    newargs = convert2pq(arg, defaultunits)
    er, a, b, d, l, freq = tuple(newargs)
    kc = (pi/ a)
    k = 2 * pi * freq * csqrt(er) / co
    Zo = csqrt(mu0 / eps0 / er) * k / np.sqrt(k * k - kc * kc) * 2. * b / a  # impedance of waveguide
    lg = 2 * pi / np.sqrt(k ** 2.0 - kc ** 2.0)  # lambda guided
    tt = (l/ d)

    def integrand(alpha):
        alpha1 = csqrt(1 - alpha ** 2.0)
        return ((scipy.special.ellipk(alpha) - alpha1 ** 2 * scipy.special.ellipe(alpha))/ (scipy.special.ellipk(alpha1) - alpha ** 2 * scipy.special.ellipe(alpha1)))
    if isinstance(tt, float):
        alpha = scipy.optimize.brentq(lambda x: integrand(x) - tt, 0.001, 0.999)
    else:
        alpha = np.array([])
        for i in tt:
            alpha = append(
                alpha, scipy.optimize.brentq(lambda x: integrand(x) - i, 0.001, 0.999,  maxiter=2000))
    alpha1 = csqrt(1 - alpha ** 2.0)
    D2 = d / np.sqrt(2) * alpha1
    D2 = D2 * ((1./ (scipy.special.ellipe(alpha1) - alpha ** 2 * scipy.special.ellipk(alpha1))))

    def integrand(alpha):
        alpha1 = csqrt(1 - alpha ** 2.0)
        return ((scipy.special.ellipk(alpha1) - alpha ** 2 * scipy.special.ellipe(alpha1))/ (scipy.special.ellipk(alpha) - alpha1 ** 2 * scipy.special.ellipe(alpha)))
    if isinstance(tt, float):
        alpha = scipy.optimize.brentq(lambda x: integrand(x) - tt, 0.001, 0.999)
    else:
        alpha = np.array([])
        for i in tt:
            alpha = np.append(
                alpha, scipy.optimize.brentq(lambda x: integrand(x) - i, 0.001, 0.999,  maxiter=2000))
    alpha1 = csqrt(1 - alpha ** 2.0)
    D1 = csqrt(np.sqrt(alpha ** 2 * alpha1 ** 2 / 3.0)) * d / \
        (scipy.special.ellipe(alpha) - alpha1 ** 2 * scipy.special.ellipk(alpha))
    Xa = Zo * 2 * a / lg * (a / pi / D2) ** 2
    Xb = Zo * a / 8 / lg * (pi * D1 / a) ** 4
    L = Xa / 2.0 / pi / freq / Zo
    C = 1.0 / 2.0 / pi / freq / Xb / Zo
    arg.append(prettystring(L, defaultunits[6]))
    arg.append(prettystring(C, defaultunits[7]))
    arg.append(prettystring(Zo, defaultunits[8]))
    return arg


def EvanescentWGEquivalent(arg, defaultunits=[]):
    """ Waveguide Width Step from Rectangular Waveguide to Evanescent Mode Rectangular Waveguide.

    Args:
        arg(list): First 5 arguments are inputs.

            1. Waveguide Width;length
            2. Waveguide Height;length
            3. Dielectric Permittivity;
            4. Waveguide Length;length
            5. Frequency; frequency
            6. Series Inductance For Shunt-Series-Shunt Model; inductance
            7. Shunt Inductance For Shunt-Series-Shunt Model; inductance
            8. Series Inductance For Series-Shunt-Series Model; inductance
            9. Shunt Inductance For Series-Shunt-Series Model; inductance
            10. Characteristic Impedance; impedance
            Reference:  The Design of Evanescent Mode Waveguide Bandpass Filters for a Prescribed Insertion Loss Characteristic.pdf
            Model= Xp1,Xs1,Xp1 ya da Xs2,Xp2,Xs2 (p: shunt, s: series)
            Zo=jXo

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:5]
    newargs = convert2pq(arg, defaultunits)
    a, b, er, length, frek = tuple(newargs)
    lcutoff = 2 * a
    wavelength = co / np.sqrt(er) / frek
    Xo = abs(Z_WG_TE10(er, a, b, frek))
    gamma = 2. * pi / wavelength * csqrt(((wavelength/ lcutoff)) ** 2.0 - 1.)
    Xs1 = Xo * np.sinh(gamma * length)
    Xs2 = Xo * np.tanh(gamma * length / 2.0)
    Xp1 = (Xo/ np.tanh(gamma * length / 2.0))
    Xp2 = (Xo/ np.sinh(gamma * length))
    arg.append(prettystring(Xs1 / 2. / pi / frek, defaultunits[5]))
    arg.append(prettystring(Xp1 / 2. / pi / frek, defaultunits[6]))
    arg.append(prettystring(Xs2 / 2. / pi / frek, defaultunits[7]))
    arg.append(prettystring(Xp2 / 2. / pi / frek, defaultunits[8]))
    arg.append(prettystring(Xo, defaultunits[9]))
    return arg


def EWG_ABCD(a, b, er, length, frek):
    # Referans:"The Design of Evanescent Mode Waveguide Bandpass Filters for a Prescribed Insertion Loss Characteristic.pdf"
    # Model= Xp1,Xs1,Xp1 ya da Xs2,Xp2,Xs2 (p: shunt, s: series)
    # Zo=jXo
    lcutoff = 2 * a
    wavelength = co / np.sqrt(er) / frek
    Xo = abs(Z_WG_TE10(er, a, b, frek))
    gamma = 2. * pi / wavelength * csqrt(((wavelength/ lcutoff)) ** 2.0 - 1.)
    Xs1 = Xo * np.sinh(gamma * length)
    Xs2 = Xo * np.tanh(gamma * length / 2.0)
    Xp1 = (Xo/ np.tanh(gamma * length / 2.0))
    Xp2 = (Xo/ np.sinh(gamma * length))
    networks = []
    networks.append(shunt_z(1.0j * Xp1))
    networks.append(series_z(1.0j * Xs1))
    networks.append(shunt_z(1.0j * Xp1))
    return CascadeNetworks(networks)


def EWG_inv(a, b, er, length, frek):
    # Referans:"The Design of Evanescent Mode Waveguide Bandpass Filters for a Prescribed Insertion Loss Characteristic.pdf"
    # Model= Xp1,Xs1,Xp1 ya da Xs2,Xp2,Xs2 (p: shunt, s: series)
    # Zo=jXo
    lcutoff = 2 * a
    wavelength = co / np.sqrt(er) / frek
    Xo = abs(Z_WG_TE10(er, a, b, frek))
    gamma = 2. * pi / wavelength * csqrt(((wavelength/ lcutoff)) ** 2.0 - 1.)
    Xs1 = Xo * np.sinh(gamma * length)
    Xs2 = Xo * np.tanh(gamma * length / 2.0)
    Xp1 = (Xo/ np.tanh(gamma * length / 2.0))
    Xp2 = (Xo/ np.sinh(gamma * length))
    networks = []
    networks.append(shunt_z(-1.0j * Xs1))
    networks.append(series_z(1.0j * Xs1))
    networks.append(shunt_z(-1.0j * Xs1))
    return CascadeNetworks(networks)


def RectWG2EvanescentRectWGStep(a1, a2):
    """ Waveguide Width Step from Rectangular Waveguide to Evanescent Mode Rectangular Waveguide.

    Args:
        arg(list): First 2 arguments are inputs.

            1. Width of Rectangular Waveguide;length;
            2. Width of Evanescent Mode Rectangular Waveguide;length;
            3. Inductance; inductance
            4. Turns Ratio;
            Reference:  The Design of Evanescent Mode Waveguide Bandpass Filters for a Prescribed Insertion Loss Characteristic.pdf

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:2]
    newargs = convert2pq(arg, defaultunits)
    a1, a2 = tuple(newargs)
    n = csqrt((a1/ a2)) * (1. - ((a2/ a1)) ** 2.) / \
        (4. / pi * np.cos(pi / 2. * a2 / a1))
    L = (a1 * mu0 * np.tan(pi / 2. * a2 / a1) ** 2. / pi)
    arg.append(prettystring(L, defaultunits[2]))
    arg.append(prettystring(n, defaultunits[3]))
    return arg


def Star2TriangleTransformation(arg, defaultunits=[]):
    """ Star network to Triangle network transformation.

    Args:
        arg(list): First 3 arguments are inputs.

            1. Z1; impedance
            2. Z2; impedance
            3. Z3; impedance
            4. Z1'; impedance
            5. Z2'; impedance
            6. Z3'; impedance
            Reference:
            At star, z1 is connected to A-node, z2 is connected to B-node, z3 is connected to C-node
            At triangle, z1 is between A-B, z2 is between A-C, z3 is between B-C

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:3]
    newargs = convert2pq(arg, defaultunits)
    z1, z2, z3 = tuple(newargs)
    a = (1./ z1) + (1./ z2) + (1./ z3)
    arg.append(prettystring(z1 * z2 * a, defaultunits[3]))
    arg.append(prettystring(z1 * z3 * a, defaultunits[4]))
    arg.append(prettystring(z2 * z3 * a, defaultunits[5]))
    return arg


def Triangle2StarTransformation(arg, defaultunits=[]):
    """ Triangle network to Star network transformation.

    Args:
        arg(list): Last 3 arguments are inputs.

            1. Z1; impedance
            2. Z2; impedance
            3. Z3; impedance
            4. Z1'; impedance
            5. Z2'; impedance
            6. Z3'; impedance
            Reference:
            At star, z1 is connected to A-node, z2 is connected to B-node, z3 is connected to C-node
            At triangle, z1' is between A-B, z2' is between A-C, z3' is between B-C

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg1 = arg[3:6]
    newargs = convert2pq(arg1, defaultunits)
    print(newargs)
    z1, z2, z3 = tuple(newargs)
    a = (z2/ z1)
    b = (z3/ z2)
    s = 1 + (1.0/ b) + 1.0 / a / b
    arg[0] = (prettystring(z1 / b / s, defaultunits[3]))
    arg[1] = (prettystring((z1/ s), defaultunits[4]))
    arg[2] = (prettystring(z1 * a / s, defaultunits[5]))
    return arg


def GyselPowerDivider(arg, defaultunits=[]):
    """ Triangle network to Star network transformation.

    Args:
        arg(list): First 6 arguments are inputs.

            1. Zo1;  impedance
            2. Zo2;  impedance
            3. Zo3;  impedance
            4. R1; impedance
            5. R2; impedance
            6. P2/P3 ratio;
            7. Z1; impedance
            8. Z2; impedance
            9. Z3; impedance
            10. Z4; impedance
            Reference:
            Zo1: 1. port impedance
            Zo2: 2. port impedance
            Zo3: 3. port impedance
            R1: 1. isolation resistor (2. porta yakin)
            R2: 2. isolation resistor (3. porta yakin)
            ratio: P2/P3 power ratio
            Z1: impedance of transmission line between 1.port and 2.port
            Z2: impedance of transmission line between 1.port and 3.port
            Z3: impedance of transmission line between 2.port and isolation resistor
            Z4: impedance of transmission line between 3.port and isolation resistor

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:6]
    newargs = convert2pq(arg, defaultunits)
    zo1, zo2, zo3, R1, R2, ratio = tuple(newargs)
    z1 = csqrt(zo1 * zo2 * (1.0 + ratio) / ratio)
    z2 = csqrt(zo1 * zo3 * (1.0 + ratio))
    z3 = csqrt(R1 * R2 * zo2 * (1.0 + ratio) / (R1 + R2))
    z4 = csqrt(R1 * R2 * zo3 * (1.0 + ratio) / (R1 + R2) / ratio)
    arg.append(prettystring(z1, defaultunits[6]))
    arg.append(prettystring(z2, defaultunits[7]))
    arg.append(prettystring(z3, defaultunits[8]))
    arg.append(prettystring(z4, defaultunits[9]))
    return arg


def DualTransformation1(arg, defaultunits=[]):
    """ Dual Transformation 1.

    Args:
        arg(list): First 4 arguments are inputs.

            1.  L1 ; inductance
            2.  C1 ; capacitance
            3.  L2 ; inductance
            4.  C2 ; capacitance
            5.  L1' ; inductance
            6.  C1' ; capacitance
            7.  L2' ; inductance
            8.  C2' ; capacitance
            Reference:  Microstrip Filters for RF-Microwave Applications, s.25, Figure 2.6a

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    arg = arg[:4]
    newargs = convert2pq(arg, defaultunits)
    L1, C1, L2, C2 = tuple(newargs)
    a = L1 * L2 * C1 * C2
    b = L1 * C1 + L2 * C2 + L2 * C1
    c = L1 * L2 * C1
    d = L2
    e = ((b + csqrt(b * b - 4 * a))/ 2.0)
    C1_ = ((a * (a - e * e))/ (e * (a * d - c * e)))
    L1_ = ((a * d - c * e)/ (a - e * e))
    C2_ = ((a - e * e)/ (c - d * e))
    L2_ = ((e * (c - d * e))/ (a - e * e))
    argout = [L1_, C1_, L2_, C2_]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg


def DualTransformation2(arg, defaultunits=[]):
    """ Dual Transformation 1.

    Args:
        arg(list): First 4 arguments are inputs.

            1.  L1 ; inductance
            2.  C1 ; capacitance
            3.  L2 ; inductance
            4.  C2 ; capacitance
            5.  L1' ; inductance
            6.  C1' ; capacitance
            7.  L2' ; inductance
            8.  C2' ; capacitance
            Reference:  Microstrip Filters for RF-Microwave Applications, s.25, Figure 2.6b

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    arg = arg[:4]
    newargs = convert2pq(arg, defaultunits)
    L1, C1, L2, C2 = tuple(newargs)
    a = L1 * L2 * C1 * C2
    b = L1 * C1 + L2 * C2 + L1 * C2
    c = L1 * C2 * C1
    d = C2
    e = ((b + csqrt(b * b - 4 * a))/ 2.0)
    L1_ = ((a * (a - e * e))/ (e * (a * d - c * e)))
    C1_ = ((a * d - c * e)/ (a - e * e))
    L2_ = ((a - e * e)/ (c - d * e))
    C2_ = ((e * (c - d * e))/ (a - e * e))
    argout = [L1_, C1_, L2_, C2_]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def thermal_conductance_of_via_farm(arg, defaultunits):
    """Thermal conductance of an array of vias in PCB.

    Args:
        arg(list): First 7 arguments are inputs.

            1. Plated Via Diameter (d);length
            2. Plating Thickness (t);length
            3. Area Width (w);length
            4. Area Height (l);length
            5. Dielectric Height (h);length
            6. Number Of Vias (n);
            7. Dielectric Thermal Conductivity ;   thermal conductivity
            8. Metal Thermal Conductivity ; thermal conductivity
            9. Thermal Conductance (W/K) ;
            10. Thermal Resistance (K/W) ;

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """

    arg = arg[:8]
    newargs = convert2pq(arg, defaultunits)
    d, t, w, l, h, n, sd, sm = tuple(newargs)
    Svia_total = n*pi*(d*t+t*t)
    Sdiel = w*l - n*pi*((d/2.0)+t)**2
    st = sd*Sdiel/h+sm*Svia_total/h
    A = (w*h-n*pi*((d/2.0))**2)
    st = st*(sm*A/t/2.0)/(st+sm*A/t/2.0)  # Alt ve ust toprak tabakalarını da kattık
    argout = [st,(1.0/st)]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def thermal_conductance_of_via_farm_view(arg, defaultunits):
    """
    """
    arg = arg[:8]
    newargs = convert2pq(arg, defaultunits)
    d, t, w, l, h, n, sd, sm = tuple(newargs)
    # visvis.clf()
    m=int(np.round(np.sqrt(n)))
    xstep=w/t/(m)
    ystep=l/t/np.ceil(n/m)
    # diel = visvis.solidBox(translation=(0,0,h/t/2.0),scaling=((w/t),(l/t),(h/t)))
    # gnd = visvis.solidBox(translation=(0,0,((h+(t/2.0))/t)),scaling=((w/t),(l/t),1.0))
    # gnd1 = visvis.solidBox(translation=(0,0,-t/t/2.0),scaling=((w/t),(l/t),1.0))
    # cyls=[]
    # cyl1s=[]
    # for j in range(1,int(n/m)+2):
    #     for i in range(1,m+1):
    #         if  ((j-1)*m+i) < (n+1):
    #           cyls.append( visvis.solidCylinder(translation=(-w/t/2.0+(i-0.5)*xstep,-l/t/2.0+(j-0.5)*ystep,-1.0-0.05),scaling=(d/t/2.0,d/t/2.0,(h+2.0*t)/t+0.1),N=64))
    #           cyl1s.append( visvis.solidCylinder(translation=(-w/t/2.0+(i-0.5)*xstep,-l/t/2.0+(j-0.5)*ystep,-1.0-0.1),scaling=(d/t/2.0-1.0,d/t/2.0-1.0,(h+2.0*t)/t+0.2),N=64))
    #           cyls[-1].faceColor="k"
    #           cyl1s[-1].faceColor="w"
    # diel.faceColor="g"
    # gnd.faceColor="r"
    # gnd1.faceColor="r"
    return

def Exponential_Taper_Impedance_Transformer(arg, defaultunits=[]):
    """ Exponential Impedance Taper.

    Args:
        arg(list): First 5 arguments are inputs.

            1.  Source Impedance ; impedance
            2.  Load Impedance ; impedance
            3.  Number Of Sections ;
            4.  Fractional Bandwidth (F2/F1) ;
            5.  Length (normalized to Lambda at fcenter) ;
            6.  Impedances ; impedance
            7.  Return Loss ;
            Reference:  Foundations for Microwave Engineering, Collin

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:5]
    newargs = convert2pq(arg, defaultunits)
    Z0, ZL, N, BW, L_lambda = tuple(newargs)
    r = (ZL/ Z0)
    N = int(np.round(N))
    print("N ", N)
    # (np.arange(float(N+1))/N), corresponds to z/L in formula.
    z_n = np.exp((np.arange(float(N + 1))/N) * np.log(r)) * Z0

    # Balta metoduyla gammanin ban icindeki en kotu degerini bulma
    L_lambda_min = L_lambda * ((2.0/ (1.0 + BW)))
    L_lambda_max = L_lambda * (2.0 * BW / (1.0 + BW))
    Ls = np.linspace(L_lambda_min, L_lambda_max, 50)
    max_gamma = ( 20.0 * np.log10(np.fabs(0.5 * np.log(r) * np.sin(2 * pi * Ls) / (2 * pi * Ls)))).max()
    arg.append(globsep2.join([prettystring(x, defaultunits[5]) for x in z_n]))
    arg.append(prettystring(max_gamma, defaultunits[6]))
    return arg


def Triangular_Taper_Impedance_Transformer(arg, defaultunits=[]):
    """ Triangular Impedance Taper.

    Args:
        arg(list): First 5 arguments are inputs.

            1.  Source Impedance ; impedance
            2.  Load Impedance ; impedance
            3.  Number Of Sections (Even) ;
            4.  Fractional Bandwidth (F2/F1) ;
            5.  Length (normalized to Lambda at fcenter) ;
            6.  Impedances ; impedance
            7.  Return Loss ;
            Reference:  Foundations for Microwave Engineering, Collin

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:5]
    newargs = convert2pq(arg, defaultunits)
    Z0, ZL, N, BW, L_lambda = tuple(newargs)
    r = (ZL/ Z0)
    N = int(np.round(N/2.0))
    print("N ", N)
    # (np.arange(float(N+1))/N), corresponds to z/L in formula.
    z_n1 = np.exp(2.0 * (np.arange(float(N))/N) ** 2 * np.log(r)) * Z0
    # (np.arange(float(N+1))/N), corresponds to z/L in formula.
    z_n2 = np.exp((2 * (np.arange(float(N))/N) + 0.5 - 2.0 *
                  (np.arange(float(N))/N) ** 2) * np.log(r)) * Z0

    # Balta metoduyla gammanin ban icindeki en kotu degerini bulma
    L_lambda_min = L_lambda * ((2.0/ (1.0 + BW)))
    L_lambda_max = L_lambda * (2.0 * BW / (1.0 + BW))
    Ls = np.linspace(L_lambda_min, L_lambda_max, 50)
    max_gamma = (20.0 * np.log10(np.fabs(0.5 * np.log(r) * ((np.sin(pi * Ls)/ (pi * Ls))) ** 2))).max()

    arg.append(globsep2.join([prettystring(x, defaultunits[5])
               for x in (list(z_n1) + list(z_n1) + [ZL])]))
    arg.append(prettystring(max_gamma, defaultunits[6]))
    return arg

def Patch_Antenna_Analysis(arg, defaultunits=[]):
    """ Calculates performance and impedance values for an N-section Chebyshev Impedance Taper.

    Args:
        arg(list): First 6 arguments are inputs.

            1.  Width (W) ; length
            2.  Length (L) ; length
            3.  Substrate Thickness (h);length
            4.  Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>) ;
            5.  Dielectric Loss Tangent ;
            6.  Metal Conductivity ; electrical conductivity
            7.  Resonance Frequency (f) ; frequency
            8.  Bandwidth ; frequency
            Reference:  Foundations for Microwave Engineering, Collin

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:6]
    newargs = convert2pq(arg, defaultunits)
    w, L, h, er, tand, cond = tuple(newargs)
    ee = (er+1)/2+(er-1)/2*sqrt(1+12*h/w)
    dL=0.412*h*(ee+0.3)*(w/h+0.264)/(ee-0.258)/(w/h+0.8)
    f = co/(2*sqrt(er)*(L+dL))
    k0 = 2*pi*f/co
    c1 = 1-1/er+2/5/er**2
    erhed = 1/(1+3*pi*k0*h*(1-1/er)**3/4/c1)
    Rs=sqrt(pi*f*mu0/cond)

    c2 = -0.0914153
    a2 = -0.16605
    a4 = 0.00761
    p = 1+0.1*a2*(k0*w)**2+(a2**2+2*a4)*(3/560)*(k0*w)**4+c2*(k0*L)**2/5+a2*c2*(k0*w)**2*(k0*L)**2/70
    Qd = 1/tand
    Qc = eta0*k0*h/2/Rs
    Qsp = 3/16*(2*pi*er*L)/(k0*p*c1*w*h)
    Qsw = Qsp*erhed/(1-erhed)
    Q = 1/(1/Qd+1/Qc+1/Qsp/erhed)
    BW = 1/sqrt(2)/Q
    argout = [f, BW]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg


def Chebyshev_Taper_Impedance_Transformer(arg, defaultunits=[]):
    """ Calculates performance and impedance values for an N-section Chebyshev Impedance Taper.

    Args:
        arg(list): First 5 arguments are inputs.

            1.  Source Impedance ; impedance
            2.  Load Impedance ; impedance
            3.  Number Of Sections (Even) ;
            4.  Fractional Bandwidth (F2/F1) ;
            5.  Length (normalized to Lambda at fcenter) ;
            6.  Impedances ; impedance
            7.  Return Loss ;
            Reference:  Foundations for Microwave Engineering, Collin

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:5]
    newargs = convert2pq(arg, defaultunits)
    Z0, ZL, N, BW, L_lambda = tuple(newargs)
    r = (ZL/ Z0)
    N = int(np.round(N/2.0))
    print("N ", N)
    p = 2 * pi * (np.arange(float(N + 1))/N) - pi
    z_n = Z0 * np.exp(np.log(r) / (2 * pi) *
                   (p + pi + 0.632 * np.sin(p) - 0.0653 * np.sin(3 * p)))

    # Finding maximum gamma in band
    L_lambda_min = L_lambda * ((2.0/ (1.0 + BW)))
    L_lambda_max = L_lambda * (2.0 * BW / (1.0 + BW))
    Ls = np.linspace(L_lambda_min, L_lambda_max, 50)
    max_gamma = ((20.0 * np.log10(np.fabs(0.5 * np.log(r) * ((np.sin(pi * Ls)/ (pi * Ls))) ** 2)))).max()

    arg.append(globsep2.join([prettystring(x, defaultunits[5]) for x in z_n]))
    arg.append(prettystring(max_gamma, defaultunits[6]))
    return arg

def fcutoff_CWG(rad,eps_r=1, v=0, n=1, mode="TE"):
    r""" Computes the cutoff frequency of circular waveguide.

    Args:
        v (int): Mode number of :math:`\phi`.
        n (int): Radial mode number.
        eps_r (float): Permittivity of filling material.
        mode (str): "TE" or "TM".
        rad (float): Radius.
    Returns:
        fc (float): Cutoff frequency (Hz).
    """
    if mode=="TE":
        kc = jnp_zeros(v,n)/rad
    else: # mode=="TM" is assumed
        kc = jn_zeros(v,n)/rad
    fc= kc*co/2/np.pi/sqrt(eps_r)
    return fc

def Z_CWG(rad,freq, eps_r=1, v=0, n=1, mode="TE"):
    r""" Computes the wave impedance of circular waveguide.

    Args:
        v (int): Mode number of :math:`\phi`.
        n (int): Radial mode number.
        eps_r (float): Permittivity of filling material.
        freq (float): Frequency (Hz).
        mode (str): "TE" or "TM".
        rad (float): Radius.
    Returns:
        Z (float): Impedance.
    """
    k=2*np.pi*freq*sqrt(eps_r)/co
    if mode=="TE":
        kc = jnp_zeros(v,n)/rad
        beta= csqrt(k**2-kc**2)
        Z= k*eta0/sqrt(eps_r)/beta
    else: # mode=="TM" is assumed
        kc = jn_zeros(v,n)/rad
        beta= csqrt(k**2-kc**2)
        Z= beta*eta0/sqrt(eps_r)/k
    return Z

def Klopfenstein_Taper_Impedance_Transformer(arg, defaultunits=[]):
    r""" Calculates performance and impedance values for an N-section Klopfenstein Impedance Taper.

    Args:
        arg(list): First 6 arguments are inputs.

            1.  Source Impedance ; impedance
            2.  Load Impedance ; impedance
            3.  Maximum Reflection Coefficient (dB) ;
            4.  Number Of Sections ;
            5.  Minimum Frequency ; frequency
            6.  Test Frequency ; frequency
            7.  Minimum Total Phase at Minimum Frequency ; angle ;
            8.  Impedances ; impedance
            9.  MAG(Reflection Coefficient) ;
            Reference:  Microwave Engineering, Pozar

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:6]
    newargs = convert2pq(arg, defaultunits)
    Z01, ZL1, G, N, fminimum, ftest = tuple(newargs)
    ZL = max([ZL1, Z01])
    Z0 = min([ZL1, Z01])

    def fi(ys, A):
        return np.array([scipy.integrate.quad(lambda x:(scipy.special.iv(1, A * csqrt(1.0 - x * x))/ (A * csqrt(1.0 - x * x))), 0, y)[0] for y in ys])

    N = int(np.round(N))
    p = (np.arange(float(N + 1))/N)    # z/L
    Gamma_0 = ((ZL - Z0)/ (ZL + Z0))
    # Using this approximation method, there is no impedance jump at two ends of the taper,
    # but there is a slight detoriation at performance at low frequency edge of the spectrum
    Gamma_0 = 0.5 * np.log((ZL/ Z0))
    Gamma_Max = np.power(10.0, (-abs(G)/ 20.0))
    A = np.arccosh((Gamma_0/ Gamma_Max))
    z_n = np.exp(0.5 * np.log(Z0 * ZL) + Gamma_0 / np.cosh(A)
                 * (A * A * fi(2.0 * p - 1.0, A)))
    if (ZL1 < Z01):
        z_n = np.flipud(z_n)  # numpy dizisini ters cevirme

def Absorptive_Filter_Equalizer(arg, defaultunits=[]):
    r""" Equalizer using an absorptive filter composed of two coupled lines.

    Args:
        arg(list): First 4 arguments are inputs.

            1.  Reference Impedance ; impedance
            2.  Coupling (dB) ;
            3.  Center Frequency ; frequency
            4.  Test Frequency ; frequency
            5.  S21 (dB) ;
            6.  Zeven ;  impedance
            7.  Zodd ;  impedance

            Reference:

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:4]
    newargs = convert2pq(arg, defaultunits)
    Z0, coupling, fcenter, ftest= tuple(newargs)
    k= np.power(10.0,(-np.fabs(coupling)/20.0))
    theta=pi/2.0*ftest/fcenter
    a=1.0j*k*np.sin(theta)/(np.sqrt(1.0-k*k)*np.cos(theta)+1.0j*np.sin(theta))
    b=(np.sqrt(1.0-k*k)/(np.sqrt(1.0-k*k)*np.cos(theta)+1.0j*np.sin(theta)))
    S21=a*a*np.np.exp(-1.0j*theta)/(1.0-b*b*np.np.exp(-2.0j*theta))
    dBS21=20.0*np.np.log10(np.abs(S21))
    Zeven,Zodd=Z0*(1.0+k)/(1.0-k),Z0*(1.0-k)/(1.0+k)
    argout = [dBS21,dBS11,Zeven,Zodd]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def LC_Balun(arg, defaultunits=[]):
    r""" Calculate LC Balun.

    Args:
        arg(list): First 4 arguments are inputs.

            1.  Source Impedance (Rin) ; impedance
            2.  Load Impedances (RL) ; impedance
            3.  Frequency; frequency
            4.  Test Frequency ; frequency
            5.  Inductance ; inductance
            6.  Capacitance ; capacitance
            7.  S11 (dB) ;
            8.  S21 (dB) ;
            9.  S31 (dB) ;
            Reference:

        defaultunits(list, optional): Default units for quantities in *arg* list. Default is [] which means SI units will be used if no unit is given in *arg*.

    Returns:
        list: arg
    """
    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 2
    arg = arg[:6]
    newargs = convert2pq(arg, defaultunits)
    Rin, RL, freq, ftest  = tuple(newargs)
    X=csqrt(Rin*2.0*RL)
    w0=2*pi*freq
    L=X/w0
    C=1.0/w0/X

    # calculate S-parameters
    XL=1.0j*X
    XC=X/1.0j
    a=1.0-Rin/(XL+XC*RL/(XC+RL))-Rin/(XC+XL*RL/(XL+RL))
    b=1.0+Rin/(XL+XC*RL/(XC+RL))+Rin/(XC+XL*RL/(XL+RL))
    b1=2.0/(XL/(XC*RL)+1.0/(XC+RL))
    b2=2.0/(XC/(XL*RL)+1.0/(XL+RL))
    S11=b/a
    S21=b1/a
    S31=b2/a
    argout = [L,C,S11,S21]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

if __name__ == "__main__":
    c = ["0.1", "4.0", "50"]
    # print SymmetricLangeCoupler([],"0.1","4.0","50")
    # print GyselPowerDivider(["50","100","50","50","50","2"],["","","","","","","","","","","","","",""])
    # print
    # Exponential_Taper_Impedance_Transformer(["50","100","20","9","2"],[""]*7)
    print(OptimumMiteredArbitraryAngleMicrostripBend(["300", "127", "90deg", "3"], ["um", "um", "deg", "um"]))
