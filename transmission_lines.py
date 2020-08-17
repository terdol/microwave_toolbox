#-*-coding:utf-8-*-

from constants import *
import sys
# from scipy import *
# from scipy.special import ellipk
#from copy import deepcopy
from genel import *
#import numpy as np
# import visvis
from numpy.lib.scimath import sqrt as csqrt
from numpy import sqrt, pi, log, log10, cos, sin, tan, cosh, sinh, tanh, exp
from numpy import arccosh, arctan, arccos, arcsin, arcsinh, arctanh, seterr
from numpy import atleast_1d,power,ndarray,array,fabs, sum, mean
co = speed_of_light_in_freespace.simplified.magnitude
eta0 = free_space_wave_impedance.simplified.magnitude
mu0 = free_space_permeability.simplified.magnitude
eps0 = free_space_permittivity.simplified.magnitude

ellipk = ekpolyfit

def physical_length(eeff, f, elec_length):
    """
    This function returns physical length in meters.

    Args:
        arg(list): First 3 arguments are inputs.

            1. eeff ( effective relative permittivity  ) ;
            2. f ; frequency
            3. elec_length (radian); angle
    """
    temp = (co * elec_length / 2 / pi / f / csqrt(eeff))
    return temp

def electrical_length(eeff, f, len):
    """
    This function returns electrical length in radians.

    Args:
        arg(list): First 3 arguments are inputs.

            1. eeff ( effective relative permittivity  ) ;
            2. f ; frequency
            3. length ; length
    """
    temp = (len * 2.0 * pi * f * csqrt(eeff) / co)
    return temp

def skin_depth(f, sigma, mu=1.0, er=0.0):
    """
    This function returns skin depth in meters.
    Ref: https://en.wikipedia.org/wiki/Skin_effect

    Args:
        arg(list): First 3 arguments are inputs.

            1. f ; frequency
            2. sigma ; conductivity
            3. mu ; relative permeability
            4. er ; relative permittivity
    """
    return csqrt(1.0 / sigma / mu / mu0 / pi / f)*csqrt(csqrt(1+(2*pi*f*er*eps0/sigma)**2)+2*pi*f*er*eps0/sigma)

def Sentez(fonk, _args, k, target_value = [],init_value = [], limits = []):
    r"""Function that is used to calculate the parameter value of a function
    that will give target value.

    Args:
        fonk (function): function to be used at optimization
        _args (list): function arguments of fonk
        k (int): list of indices of variables to be calculated by optimization
        target_value (list, optional): target value of fonk. Defaults to [].
        init_value (list, optional): initial values of variables]. Defaults to [].
        limits (list, optional): constraints on variables. Defaults to [].

    Returns:
        float: Calculated parameter
    """
    # if isinstance(k,int):
    #     k=[k]
    k = atleast_1d(k)
    if not isinstance(k,(list, ndarray)):
        print("k variable should be a list or integer")
        return
    if not isinstance(init_value, collections.Iterable):
        init_value=[init_value]*len(k)
    elif len(init_value)==1:
        init_value=init_value*len(k)
    if len(limits)==1:
        limits=limits*len(k)
    k.sort()
    def callable_func(x, grad=0):
        tempargs=[]
        tempindex=0
        for i in range(len(k)):
            tempargs=tempargs+_args[tempindex:k[i]]
            tempargs=tempargs+[x[i]]
            tempindex=k[i]+1
        tempargs=tempargs+_args[len(tempargs):]
        tempout=fonk(*tuple(tempargs))
        if not isinstance(tempout, collections.Iterable):
            tempout=(tempout,)
        out = sum([fabs(target_value[count]-tempout[count])**2 for count in range(len(k))])
        # print(out)
        return out
    # from scipy.optimize import fmin_l_bfgs_b, fmin_tnc, minimize
    # output = fmin_l_bfgs_b(callable_func,array(init_value),approx_grad=1,factr=0.01,bounds=limits,maxfun=100000,pgtol=1.0e-10, epsilon=1.0e-8)

    import nlopt
    opt = nlopt.opt(nlopt.GN_ESCH, len(k))
    opt.set_min_objective(callable_func)
    opt.set_lower_bounds([a[0] for a in limits])
    opt.set_upper_bounds([a[1] for a in limits])
    # opt.set_maxeval(1000)
    opt.set_maxtime(1)
    # opt.set_xtol_abs(0.1)
    xopt = opt.optimize(init_value)
    return xopt

def skindepth_analysis(arg, defaultunits):
    """

    Args:
        arg(list): First 3 arguments are inputs.

            1. Metal Conductivity ; electrical conductivity
            2. Metal Permeability ;
            3. Frequency ; frequency
            4. Skin Depth ;length
            5. Surface Impedance ; impedance
    """
    arg = arg[:3]
    newargs = convert2pq(arg, defaultunits)
    sigma, mu, freq = tuple(newargs)
    skindepth = skin_depth(freq, sigma, mu)
    arg.append(prettystring(skindepth, defaultunits[3]))
    arg.append(prettystring((1.0 / skindepth / sigma), defaultunits[4]))
    return arg


def Z_qs_thin_microstrip(w, h, er):
    r"""Impedance of microstrip transmission line with infinitely thin metal and ignoring dispersion.
    Reference:  Qucs Technical.pdf, Hammerstad and Jensen (er should be eeff in 11.5 formula )
    0.01% for w/h<1, 0.01% for w/h<1000

    Args:
        w (float): Line width (in m).
        h (float): Thickness of the substrate (in m).
        er (float): Dielectric permittivity of the substrate.

    Returns:
        float: Characteristic impedance.
    """
    r = (h/ w)
    fu = 6. + (2 * pi - 6.) * exp(-(30.666 * r) ** (0.7528))
    eeff = er_eff_qs_thin_microstrip(w, h, er)
    return eta0 / (2 * pi * csqrt(eeff)) * log(fu * r + csqrt(1.0+(2.0*r)** 2))


def er_eff_qs_thin_microstrip(w, h, er):
    r"""Effective dielectric permittivity of microstrip transmission line with infinitely thin metal and ignoring dispersion.
    Reference:  Hammerstad and Jensen, 0.2% for 0.01<w/h<1000 and er<128

    Args:
        w (float): Line width (in m).
        h (float): Thickness of the substrate (in m).
        er (float): Dielectric permittivity of the substrate.

    Returns:
        float: Effective dielectric permittivity.
    """
    u = (w/ h)
    b = 0.564 * pow(((er - 0.9)/ (er + 3.)), 0.053)
    a = 1. + 1. / 49. * log(((u ** 4 + ((u/ 52.)) ** 2)/
                    (u ** 4 + 0.432))) + 1. / 18.7 * log(1. + ((u/ 18.1)) ** 3)
    return ((er + 1.)/ 2.) + (er - 1.) / 2. * (1. + (10./ u)) ** (-a * b)


def Z_qs_thick_microstrip(w, h, er, t=0):
    r"""Impedance of microstrip transmission line ignoring dispersion.
    Reference:  Hammerstad and Jensen

    Args:
        w (float): Line width (in m).
        h (float): Thickness of the substrate (in m).
        er (float): Dielectric permittivity of the substrate.
        t (float. optional): Thickness of metal. Default is 0.

    Returns:
        float: Characteristic impedance.
    """
    th = (t/ h)
    wt = (w/ t)
    # print("tw= "+str(t)+"  "+str(w))
    # dw1 = (t > 0) * t / pi * \
    #     log(4. * exp(1) / csqrt((th) ** 2 + 1 / pi / (wt + 1.1)))
    # dwr = (t > 0) * 0.5 * dw1 * (1. + (1./ er))
    dw1 = t / pi * log(4. * exp(1) / csqrt((th) ** 2 + 1 / pi / (wt + 1.1)))
    dwr = 0.5 * dw1 * (1. + (1./ er))
#    w1=w+dw1
    wr = w + dwr
    x = er_eff_qs_thin_microstrip(wr, h, er)
    return (Z_qs_thin_microstrip(wr, h, 1)/ csqrt(x))


def er_eff_qs_thick_microstrip(w, h, er, t=0.0):
    """
    Ref: (Hammerstad and Jensen denenecek), Wheeler,  Qucs Technical, s.150
    """
    th = (t/ h)
    dw1=0.0
    dwr=0.0
    if (t>(h/100.0)):
        wt = (w/ t)
        dw1 = (t > 0) * t / pi * \
            log(4. * exp(1) / csqrt((th) ** 2 + 1 / pi / (wt + 1.1)))
        dwr = 0.5 * dw1 * (1. + (1./ er))
    w1 = w + dw1
    wr = w + dwr
    x = er_eff_qs_thin_microstrip(wr, h, er)
    return x * ((Z_qs_thick_microstrip(w1, h, 1.0, t)/ Z_qs_thick_microstrip(wr, h, 1.0, t))) ** 2


def er_eff_disp_thick_microstrip(w, h, t, er, f):
    """
    Ref: Kobayashi, %0.6 accuracy, 0.1<w/h<10, 1<er<128, no frequency limit
    """
    if er == 1.:
        return 1.
    ee = er_eff_qs_thick_microstrip(w, h, er, t)
    f50 = co / (2 *  pi * h * (0.75 + (0.75 - (0.332/ er ** 1.73)) * w / h)) * \
        arctan(er * csqrt(((ee - 1)/ (er - ee)))) / csqrt(er - ee)
    mc = 0
    x = (f/ f50)
    u = (w/ h)
    mc = (u/ u)  # u ile ayni boyutta 1'lerden olusan bir array yapmak icin
    mc = mc + (u < 0.7) * 1.4 / (1 + u) * (0.15 - 0.235 * exp(-0.45 * x))
    mo = 1 + (1/ (1 + csqrt(u))) + 0.32 * ((1/ (1 + csqrt(u)))) ** 3
    m = mo * mc
    return er - ((er - ee)/ (1 + x ** m))


def Z_disp_thick_microstrip(w, h, t, er, f):
    #?
    eeff = er_eff_disp_thick_microstrip(w, h, t, er, f)
    return (Z_qs_thick_microstrip(w, h, 1, t)/ csqrt(eeff))

def Z_eeff_disp_thick_microstrip(w, h, t, er, f):
    """ This function is for convenience only. Returns (Z,eeff) pair to be used at ABCD matrix of TL """
    eeff = er_eff_disp_thick_microstrip(w, h, t, er, f)
    return (Z_qs_thick_microstrip(w, h, 1, t)/ csqrt(eeff), eeff)

def average_power_rating_thick_microstrip(w, h, t, er, f, tand, sigma, mu_r, rms_roughness, Kd, dT_allowed):
    """
    Ref: Average power-handling capability of microstrip lines.pdf
    Kd: Thermal conductivity of dielectric (W/mK)
    Ka: Thermal conductivity of air (W/mK)
    dT_allowed: Maximum allowed temperature difference between line and ground
    """

    Ka = 0.026
    er_t = (Kd/ Ka)
    eeff_t = er_eff_disp_thick_microstrip(w, h, t, er_t, f)
    Z_t = (Z_qs_thick_microstrip(w, h, 1, t)/ csqrt(eeff_t))
    eeff = er_eff_disp_thick_microstrip(w, h, t, er, f)
    Z = (Z_qs_thick_microstrip(w, h, 1, t)/ csqrt(eeff))
    We = eta0 * h / (Z_t * csqrt(eeff_t))
    Weff = eta0 * h / (Z * csqrt(eeff))
    fp = Z / 2.0 / mu0 / h
    f_fp = (f/ fp)
    Weff = w + ((Weff - w)/ (1.0 + power(f_fp, 2)))
    alpha_d = dielectric_loss(eeff, er, f, tand)
    alpha_c = conductor_loss_microstrip(
        w, h, t, er, sigma, mu_r, rms_roughness, f)
    # Bu iki dPc ve dPd ifadesi 1W giri� g�c� i�in 1m'de kaybolan g�� oluyor.
    # Bunlar�n yerine birim uzunlukta kaybolan g�� olan dP/dx=0.2303 * alpha_c yazsak daha iyi olmaz m�?
    dPc = 1.0 - exp(-0.2303 * alpha_c)
    dPd = 1.0 - exp(-0.2303 * alpha_d)

    # Temperature difference between line and ground for 1W input power
    dT = h / Kd * ((dPc/ We) + dPd / 2.0 / Weff)
    return (dT_allowed/ dT)

def dc_current_rating_thick_microstrip(w, h, t, er, f, tand, sigma, mu_r, rms_roughness, Kd, dT_allowed):
    """
    Ref: Average power-handling capability of microstrip lines.pdf
    Kd: Thermal conductivity of dielectric (W/mK)
    Ka: Thermal conductivity of air (W/mK)
    dT_allowed: Maximum allowed temperature difference between line and ground
    """
    Ka = 0.026
    er_t = (Kd/ Ka)
    eeff_t = er_eff_disp_thick_microstrip(w, h, t, er_t, f)
    Z_t = (Z_qs_thick_microstrip(w, h, 1, t)/ csqrt(eeff_t))
    eeff = er_eff_disp_thick_microstrip(w, h, t, er, f)
    Z = (Z_qs_thick_microstrip(w, h, 1, t)/ csqrt(eeff))
    We = eta0 * h / (Z_t * csqrt(eeff_t))
    Weff = eta0 * h / (Z * csqrt(eeff))
    fp = Z / 2.0 / mu0 / h
    f_fp = (f/ fp)
    Weff = w + ((Weff - w)/ (1.0 + power(f_fp, 2)))
    alpha_d = dielectric_loss(eeff, er, f, tand)
    alpha_c = conductor_loss_microstrip(
        w, h, t, er, sigma, mu_r, rms_roughness, f)
    # dPc ve dPd expressions are power lost in 1m, for 1W input power
    # It might be  better to write dP/dx=0.2303 * alpha_c as power lost in unit distance
    dPc = 1.0 - exp(-0.2303 * alpha_c)
    dPd = 1.0 - exp(-0.2303 * alpha_d)

    # Temperature difference between line and ground for 1W input power
    # dT = h / Kd * (dPc / We + dPd / 2.0 / Weff)
    Imax = csqrt( Kd * We * dT_allowed * sigma * w * t / h )
    return Imax

def dielectric_loss(eeff, er, f, tand):
    """
    Gives dielectric loss in dB/m. Generic for all types of inhomogeneous transmission lines.
    Ref: Qucs Technical s.155
    """
    return (er / csqrt(eeff) * (eeff - 1) / (er - 1) * pi * f / co * tand) * Np2dB


def conductor_loss_microstrip(w, h, t, er, sigma, mu_r, rms_roughness, f):
    """
    Microstrip conductor loss as dB/m
    Ref: Qucs technical.pdf, "Conductor losses", Eq. 11.80-11.83
    """
    skindepth = skin_depth(f, sigma, mu_r)
    R = 1 / skindepth / sigma
    Kr = 1.0 + 2.0 / pi * arctan( 1.4 * ((rms_roughness/ skindepth)) ** 2) # correction for surface roughness
    ZL = Z_disp_thick_microstrip(w, h, t, er, f)
    Ki = exp(-1.2 * ((ZL/ eta0)) ** 0.7)
    return (R * Kr * Ki / ZL / w) * Np2dB


def cutoff_frequency_for_TE1_mode_microstrip(er, h):
    """
    Ref: Transmssion Line Design Handbook, p99
    """
    try:
      return co / 4.0 / h / csqrt(er - 1.0)
    #   return co * arctan(er) / (sqrt(2) * pi * h * csqrt(er - 1.0))
    except ZeroDivisionError:
      return Inf


def transverse_resonance_frequency_microstrip(er, h, w):
    r"""Transverse resonance frequency for microstrip.
    Ref: Microwave Engineering Using Microstrip Circuits, p87

    Args:
        er (float): Relative Dielectric Permittivity.
        h (float): Substrate thickness.
        w (float): Width of trace.

    Returns:
        float: Frequency.
    """
    return co / 2.0 / (h + w) / csqrt(er)
    # return co / ((0.8*h + 2*w) * csqrt(er))


def freq_limit_for_coupling_to_surface_modes_microstrip(er, h):
    r"""Minimum frequency for surface wave modes to generate.
    Ref: Microwave Engineering Using Microstrip Circuits, p86

    Args:
        er (float): Relative Dielectric Permittivity.
        h (float): Substrate thickness.
    Returns:
        float: Frequency.
    """
    return co * arctan(er) / pi / h / csqrt(2 * (er - 1))


def Z_disp_thick_covered_microstrip(w, h, h2, t, er, f):
    # Transmssion Line Design Handbook, p120  ?
    # w1=w+t/pi*(1+log(4.0)-0.5*log((t/h)**2+(t/pi/w)**2))
    # p=270.0*(1.0-tanh(1.192+0.706*csqrt(1.0-h2/h)))
    # return  Z_qs_thin(w,h,1)/csqrt( er_eff_disp_thick(w,h,t,er,f))

    """
    Ref: Lumped elements for RF and Microwave circuits, p438
    """
    p = 270.0 * (1 - tanh(0.28 + 1.2 * csqrt((h2/ h))))
    dZ = 0.0
    x = (w/ h)
    if x > 1.0:
        q = 1 - arctanh(0.48 * csqrt(x - 1.0) / (1 + (h2/ h)) ** 2)
        dZ = p * q
    else:
        dZ = p
    return ((Z_disp_thick_microstrip(w, h, t, 1.0, f) - dZ)/ csqrt(er_eff_disp_thick_covered_microstrip(w, h, h2, t, er, f)))


def er_eff_disp_thick_covered_microstrip(w, h, h2, t, er, f):
    r"""Effective dielectric permittivity of microstrip transmission line with a metallic cover.
    Reference:  Lumped elements for RF and Microwave circuits, p438

    Args:
        w (float): Line width (in m).
        h (float): Thickness of the substrate (in m).
        h2 (float): Height above the substrate up to the cover (in m).?
        t (float): Thickness of the metal (in m).
        er (float): Dielectric permittivity of the substrate.
        f (float): Frequency (in Hz).

    Returns:
        float: Effective dielectric permittivity.
    """
    F = 0.0
    x = (w/ h)
    if x > 1.0:
        F = (1.0/ csqrt(1 + (12.0/ x)))
    else:
        F = (1.0/ csqrt(1 + (12.0/ x))) + 0.041 * (1 - x) ** 2
    qf = F * tanh(1.043 + 0.121 * h2 / h - 1.164 * h / h2)
    return ((er + 1.)/ 2.0) + qf * (er - 1.) / 2.0


def microstrip_synthesis(arg, defaultunits):
    """Synthesis function for microstrip transmission lines.

    Args:
        arg(list): First 13 arguments are inputs.

            1. Line Width ;length
            2. Substrate Thickness ;length
            3. Metal Thickness ;length
            4. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            5. Dielectric Loss Tangent ;
            6. Dielectric Thermal Conductivity ;   thermal conductivity
            7. Metal Conductivity ; electrical conductivity
            8. Metal Permeability ;
            9. Roughness ;length
            10. Frequency ; frequency
            11. Physical Length ;length
            12. Impedance ;   impedance
            13. Electrical Length ;  angle
            14. Max Temp Difference (<sup>o</sup>C) ;
            15. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            16. Conductor Loss ;  loss per length
            17. Dielectric Loss ; loss per length
            18. Skin Depth ;length
            19. Cutoff Frequency for TE1 mode ; frequency
            20. Transverse Resonance Frequency; frequency
            21. Frequency Limit for Coupling to Surface Modes ; frequency
            22. Time Delay ; time
            23. L  per unit length ;
            24. C per unit length ;
            25. Surface Impedance ; impedance
            26. Average Rated Power ; power
            27. Max DC Current ; current
    """

    if len(defaultunits) == 0:
        defaultunits = [""] * len(arg) * 3
    arg = arg[:14]
    newargs = convert2pq(arg, defaultunits)
    _, h, t, er, tand, Kd, sigma, mu, roughness, freq, _, Z, rad, dT = tuple(
        newargs)
    w = h
    # print a,h,t,er,tand,Kd,sigma,mu,roughness,freq,a,Z,deg
    output = Sentez(Z_disp_thick_microstrip, [w, h, t, er, freq], [0], [Z] , [h] , [((h/1000.0),1000.0*h)])
    w=output[0]
    # print "w= ",w, type(w)
    eeff = er_eff_disp_thick_microstrip(w, h, t, er, freq)
    length = physical_length(eeff, freq, rad)
    cond_loss = conductor_loss_microstrip(
        w, h, t, er, sigma, mu, roughness, freq)
    Pave = average_power_rating_thick_microstrip(
        w, h, t, er, freq, tand, sigma, mu, roughness, Kd, dT)
    skindepth = skin_depth(freq, sigma, mu)
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    f_te1 = cutoff_frequency_for_TE1_mode_microstrip(er, h)
    f_transres = transverse_resonance_frequency_microstrip(er, h, w)
    f_limit = freq_limit_for_coupling_to_surface_modes_microstrip(er, h)
    I = dc_current_rating_thick_microstrip(w, h, t, er, freq, tand, sigma, mu, roughness, Kd, dT)
    arg[0] = prettystring(w, defaultunits[0])
    arg[10] = prettystring(length, defaultunits[10])
    argout = [eeff, cond_loss, diel_loss, skindepth, f_te1, f_transres, f_limit, (length/ ( (co/
              csqrt(eeff)) )), csqrt(eeff) * Z / co, csqrt(eeff) / Z / co, 1.0 / skindepth / sigma, Pave, I]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def microstrip_analysis(arg, defaultunits):
    r"""Analysis function for microstrip transmission lines.

    Args:
        arg(list): First 11 arguments are inputs.

            1. Line Width (w);length
            2. Substrate Thickness (h);length
            3. Metal Thickness (t);length
            4. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            5. Dielectric Loss Tangent ;
            6. Dielectric Thermal Conductivity ;   thermal conductivity
            7. Metal Conductivity ; electrical conductivity
            8. Metal Permeability ;
            9. Roughness ;length
            10. Frequency ; frequency
            11. Physical Length ;length
            12. Impedance ;   impedance
            13. Electrical Length ;  angle
            14. Max Temp Difference (<sup>o</sup>C) ;
            15. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            16. Conductor Loss ;  loss per length
            17. Dielectric Loss ; loss per length
            18. Skin Depth ;length
            19. Cutoff Frequency for TE1 mode ; frequency
            20. Transverse Resonance Frequency; frequency
            21. Frequency Limit for Coupling to Surface Modes ; frequency
            22. Time Delay ; time
            23. L  per unit length ;
            24. C per unit length ;
            25. Surface Impedance ; impedance
            26. Average Rated Power ; power
            27. Max DC Current ; current
    """

    dT = convert2pq(arg[13],defaultunits[13])[0]
    arg = arg[:11]
    newargs = convert2pq(arg, defaultunits)
    w, h, t, er, tand, Kd, sigma, mu, roughness, freq, length = tuple(newargs)
    eeff = er_eff_disp_thick_microstrip(w, h, t, er, freq)
    Z = Z_disp_thick_microstrip(w, h, t, er, freq)
    deg = electrical_length(eeff, freq, length)
    cond_loss = conductor_loss_microstrip(
        w, h, t, er, sigma, mu, roughness, freq)
    Pave = average_power_rating_thick_microstrip(
        w, h, t, er, freq, tand, sigma, mu, roughness, Kd, dT)
    skindepth = skin_depth(freq, sigma, mu)
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    f_te1 = cutoff_frequency_for_TE1_mode_microstrip(er, h)
    f_transres = transverse_resonance_frequency_microstrip(er, h, w)
    f_limit = freq_limit_for_coupling_to_surface_modes_microstrip(er, h)
    I = dc_current_rating_thick_microstrip(w, h, t, er, freq, tand, sigma, mu, roughness, Kd, dT)
    argout = [Z, deg, dT, eeff, cond_loss, diel_loss, skindepth, f_te1, f_transres, f_limit, (length/
              ( (co/ csqrt(eeff)) )), csqrt(eeff) * Z / co, csqrt(eeff) / Z / co, 1.0 / skindepth / sigma, Pave, I]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg



def microstrip_analysis_view(arg, defaultunits):
    """
    """
    # arg = arg[:11]
    # newargs = convert2pq(arg, defaultunits)
    # w, h, t, er, tand, Kd, sigma, mu, roughness, freq, length = tuple(newargs)
    # visvis.clf()
    # diel = visvis.solidBox(translation=(0,0,h/t/2.0),scaling=(w/t*5.0,w/t*10.0,(h/t)))
    # line = visvis.solidBox(translation=(0,0,(h/t)+0.5),scaling=((w/t),w/t*10.0,(t/t)))
    # gnd = visvis.solidBox(translation=(0,0,-0.5),scaling=(w/t*5.0,w/t*10.0,(t/t)))
    # diel.faceColor="g"
    # line.faceColor="r"
    # gnd.faceColor="r"
    return

def C_R_interdigital_capacitor(w,s,h,t,length,N,er,sigma, freq):
    r"""Approximate Capacitance of Interdigital Capacitor.
    Ref: RF and Microwave Coupled-Line Circuits

    Args:
        w (float): Width of fingers
        s (float): Gap between fingers
        h (float): Height of substrate
        t (float): Metal thickness
        length (float): Length of fingers
        N (int): Number of fingers
        er (float): Dielectric permittivity
        sigma (float): Electrical conductivity of metal
        freq (float): Frequency

    Returns:
        float: Capacitance.
    """
    a=(w/2.0)
    b=((w+s)/2.0)
    k=tan(((a*pi)/(4.0*b)))**2
    eeff = er_eff_qs_thick_microstrip(w, h, er, t)
    k_ = csqrt(1.0 - k ** 2)

    sd = skin_depth(freq, sigma)
    Rsurf= 1.0/sd/sigma
    Rs=4.0/3.0*length/w/N*Rsurf
    return (2.0*epso*eeff*ellipk(k)/ellipk(k_)*(N-1)*length,Rs)

def Z_thick_stripline(w, b, t, er):
    r"""Characteristic impedance of symmetric stripline transmission line.
    Reference:  Transmssion Line Design Handbook, p. 125

    Args:
        w (float): Line width (in m).
        b (float): Thickness of the substrate (in m).
        t (float): Thickness of the metal (in m).
        er (float): Dielectric permittivity of the substrate.

    Returns:
        float: Characteristic impedance.
    """

    seterr(all='raise')
    try:
        tw = (t/ w)
        Zk1 = w / 2.0 * \
            (1.0 + tw /  pi *
             (1.0 + log(4 *  pi / tw) + 0.51 *  pi * (tw) ** 2))
    except (FloatingPointError, ZeroDivisionError):
        Zk1 = (w/ 2.0)
    Zo1 = 60.0 / csqrt(er) * log(4 * b /  pi / Zk1)

    tb = (t/ b)
    wb = (w/ b)
    try:
        Zk2 = 2.0 / (1.0 - tb) * log((1/ (1 - tb)) + 1) - \
            ((1/ (1 - tb)) - 1) * log((1/ (1 - tb) ** 2) - 1)
    except:
        Zk2 = 2.0 * log(2.0)
    Zo2 = 94.15 / csqrt(er) / ((wb/ (1 - tb)) + (Zk2/  pi))
    if wb>0.35:
        return Zo2
    else:
        return Zo1

def Z_thick_offset_stripline(w, eps_r, h1, h2, t):
    """Characteristic impedance of asymmetric stripline transmission line.
    Ref: Transmssion Line Design Handbook, p. 129

    Args:
        w (float): Line width (in m).
        eps_r (float): Dielectric permittivity of the substrate.
        h1 (float): Thickness of the substrate under the line (in m).
        h2 (float): Thickness of the substrate above the line (in m).
        t (float): Thickness of the metal (in m).

    Returns:
        float: Characteristic impedance.
    """
    # print(w, eps_r, h1, h2, t)
    seterr(all='raise')
    def F(x):
        return (1-2*x)*((1-x)*log(1-x)-x*log(x))
    b = h1+h2+t
    s= fabs(h1-h2)
    eeff = eps_r
    cl=(b-s)/2
    if w/(b-t)<0.35:
        x=min(t,w)/max(w,t)
        d0 = w*(0.5008+1.0235*x-1.023*x**2+1.1564*x**3-0.4749*x**4)
        A=sin(pi*cl/b)/tanh(pi*d0/2/b)
        Z_0 = eta0*arccosh(A)/2/pi/sqrt(eps_r)
    else:
        if w/(b-t)<t/b:
            k = 1.0/cosh(pi*w/2/b)
            k_ = tanh(pi*w/2/b)
            w_b = w/b+(1-t/b)**8*(ekpolyfit(k_)/ekpolyfit(k)-2/pi*log(2)-w/b)
        else:
            w_b = w/b
        beta = 1-t/b
        gamma = cl/b-t/2/b
        cf=eps_r*eps0/pi*(2*log(1/gamma/(beta-gamma))+1/gamma/(beta-gamma)*(F(t/b)-F(cl/b)))
        Z_0=eta0/sqrt(eps_r)/(w_b/gamma+w_b/(beta-gamma)+2*cf/eps_r/eps0)
    return Z_0

def conductor_loss_stripline(w, b, t, er, f, sigma, mu):
    """Calculation of conductor loss of stripline with incremental inductance rule.

    Args:
        w (float): Width of line (in m).
        b (float): Thickness of the substrate (in m).
        t (float): Thickness of the metal trace (in m).
        er (float): Dielectric permittivity.
        f (float): Frequency (in Hz).
        sigma (float): Electrical conductivity of metal trace.
        mu (float): Magnetic permeability of metal trace.

    Returns:
        float: Conductor loss in dB/m.
    """
    sd = skin_depth(f, sigma, mu)
    z1 = Z_thick_stripline(w - sd, b + sd, t - sd, 1.0)
    z2 = Z_thick_stripline(w, b, t, 1.0)
    z = Z_thick_stripline(w, b, t, er)
    return ( - pi * f / co * (z1 - z2)) /z  * 20.0 * log10(exp(1))


def stripline_analysis(arg, defaultunits):
    """Analysis function for stripline transmission lines.

    Args:
        arg(list): First 10 arguments are inputs.

            1. Line Width (w);length
            2. Ground Separation (h);length
            3. Metal Thickness (t);length
            4. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            5. Dielectric Loss Tangent ;
            6. Metal Conductivity ; electrical conductivity
            7. Metal Permeability ;
            8. Roughness ;length
            9. Frequency ; frequency
            10. Physical Length ;length
            11. Impedance ;   impedance
            12. Electrical Length ;   angle
            13. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            14. Conductor Loss ;   loss per length
            15. Dielectric Loss ;   loss per length
    """

    arg = arg[:10]
    newargs = convert2pq(arg, defaultunits)
    w, b, t, er, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    eeff = er
    Z = Z_thick_stripline(w, b, t, er)
    deg = electrical_length(eeff, freq, length)
    cond_loss = conductor_loss_stripline(w, b, t, er, freq, sigma, mu)
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    argout = [Z, deg, eeff, cond_loss, diel_loss]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def stripline_analysis_view(arg, defaultunits):
    """
    """
    arg = arg[:10]
    newargs = convert2pq(arg, defaultunits)
    w, b, t, er, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    # visvis.clf()
    # diel = visvis.solidBox(translation=(0,0,b/t/2.0),scaling=(w/t*5.0,w/t*10.0,(b/t)))
    # line = visvis.solidBox(translation=(0,0,b/t/2.0),scaling=((w/t),w/t*10.1,(t/t)))
    # gnd1 = visvis.solidBox(translation=(0,0,-0.5),scaling=(w/t*5.0,w/t*10.0,(t/t)))
    # gnd2 = visvis.solidBox(translation=(0,0,(b/t)+0.5),scaling=(w/t*5.0,w/t*10.0,(t/t)))
    # diel.faceColor="g"
    # line.faceColor="r"
    # gnd1.faceColor="r"
    # gnd2.faceColor="r"
    return

def stripline_synthesis(arg, defaultunits):
    """Synthesis function for stripline transmission line.

    Args:
        arg(list): First 10 arguments are inputs.

            1. Line Width ;length
            2. Ground Separation ;length
            3. Metal Thickness ;length
            4. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            5. Dielectric Loss Tangent ;
            6. Metal Conductivity ; electrical conductivity
            7. Metal Permeability ;
            8. Roughness ;length
            9. Frequency ; frequency
            10. Physical Length ;length
            11. Impedance ;   impedance
            12. Electrical Length ;   angle
            13. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            14. Conductor Loss ;   loss per length
            15. Dielectric Loss ;   loss per length
    """
    arg = arg[:12]
    newargs = convert2pq(arg, defaultunits)
    w, b, t, er, tand, sigma, mu, roughness, freq, length, Zc, elec_length = tuple(
        newargs)
    eeff = er
    w = (b/ 5.0)
    output = Sentez(Z_thick_stripline, [w, b, t, er], [0], [Zc] , [b] , [((b/1000.0),1000.0*b)])
    w=output[0]
    print("ee ", eeff, freq, elec_length)
    length = physical_length(eeff, freq, elec_length)
    cond_loss = conductor_loss_stripline(w, b, t, er, freq, sigma, mu)
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    arg[0] = prettystring(w, defaultunits[0])
    arg[9] = prettystring(length, defaultunits[9])
    argout = [eeff, cond_loss, diel_loss]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def Z_coaxial(er, r, d):
    return eta0 * log((d/ r)) / 2.0 / pi / csqrt(er)

def Z_coaxial_strip_center(er, w, D):
    z = eta0 * log(2.0 * D / w) / 2.0 / pi / csqrt(er)
    if z < 300.0 * pi / csqrt(er):
        z = 15.0 * pi * pi / \
            csqrt(er) * 1.0 / (log(2.0 * (D + w) / (D - w)))
    return z

def Z_square_coaxial(er, r, d):
    z = eta0 * log(1.0787 * d / r) / 2.0 / pi / csqrt(er)
    if z < 2.0:
        z = 21.2 * csqrt((d/ r) - 1.0)
    return z

def Z_square_coaxial_square_center(er, r, d):
    z = eta0 * ((1.0/ (4.0 * ((2.0/ ((d/ r) - 1.0)) + 0.558)))) / csqrt(er)
    return z

def Z_eccentric_coaxial(er, r, d, sh):
    return eta0 * arccosh(0.5 * d / r * (1.0 - ((sh/ d)) ** 2) + 0.5 * r / d) / 2.0 / pi / csqrt(er)

def Z_parallel_wires(er, d1, d2, D):
    return eta0 * arccosh(((4.0 * D ** 2 - d1 ** 2 - d2 ** 2)/ (2.0 * d1 * d2))) / 2.0 / pi / csqrt(er)

def conductor_loss_coaxial(er, r, d, f, sigma, mu):
    r"""Conductor loss of coaxial transmission line.
    Ref: http://www.microwaves101.com/encyclopedia/coax_exact.cfm

    Args:
        er (float): Relative dielectric permittivity
        r (float): Inner radius.
        d (float): Outer radius.
        f (float): Frequency.
        sigma (float): Electrical conductivity.
        mu (float): Relative magnetic permeability.

    Returns:
        float: Conductor loss in dB/m.
    """
    # Transmssion Line Design Handbook, p47, r-inner diameter, d-outer diameter
    # return 0.014272*csqrt(f)*(1/r+1/d)/ Z_coaxial(er,r,d)
    # r: inner radius, d: outer radius
    # inner and outer conductors are assumed to be the same
    skindepth = skin_depth(f, sigma, mu)
    print(skindepth)
    # inner conductor equivalent area
    inner_area = 2 * pi * skindepth * (r + skindepth * (1.0 * exp((-r/ skindepth)) - 1))
    print(inner_area)
    # outer conductor equivalent area
    outer_area = 2 * pi * skindepth * (d + skindepth)
    print(outer_area)
    # eger en dis konektorun dis yaricapini da hesaba katarsak
    # outer_area=2*pi*skindepth*(d+skindepth-(c+skindepth)*exp((d-c)/skindepth))
    temp = Np2dB / sigma * ((1.0/ inner_area) + (1.0/ outer_area)) / (2 * Z_coaxial(er, r, d))
    return temp

def conductor_loss_eccentric_coaxial(er, r, d, sh, f, sigma, mu):
    r"""Conductor loss of eccentric coaxial transmission line.
    Ref: Transmssion Line Design Handbook, p56, problemli? t=0 olursa ne olacak?

    Args:
        er (float): Relative dielectric permittivity
        r (float): Inner radius.
        d (float): Outer radius.
        sh (float): Offset of inner conductor from center.
        f (float): Frequency.
        sigma (float): Electrical conductivity.
        mu (float): Relative magnetic permeability.

    Returns:
        float: Conductor loss in dB/m.
    """
    # Transmssion Line Design Handbook, p47, r-inner diameter, d-outer diameter
    # return 0.014272*csqrt(f)*(1/r+1/d)/ Z_coaxial(er,r,d)
    # r: inner radius, d: outer radius
    # inner and outer conductors are assumed to be the same
    skindepth = skin_depth(f, sigma, mu)
    # inner conductor equivalent area
    inner_area = 2 * pi * skindepth * (r + skindepth * (1.0 * exp((-r/ skindepth)) - 1))
    # outer conductor equivalent area
    outer_area = 2 * pi * skindepth * (d + skindepth)
    # eger en dis konektorun dis yaricapini da hesaba katarsak
    # outer_area=2*pi*skindepth*(d+skindepth-(c+skindepth)*exp((d-c)/skindepth))
    temp = Np2dB / sigma * ((1.0/ inner_area) + (1.0/ outer_area)) / (2 * Z_coaxial(er, r, d))
#    temp=temp+10.0*log10(1.0+)
    return temp

def coaxial_line_analysis(arg, defaultunits):
    r"""Analysis function for coaxial transmission line.
    Ref: Transmssion Line Design Handbook, p47, r-inner diameter, d-outer diameter

    Args:
        arg(list): First 9 arguments are inputs.

            1. Inner Radius (r);length
            2. Outer Radius (d);length
            3. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            4. Dielectric Loss Tangent ;
            5. Metal Conductivity ;  electrical conductivity
            6. Metal Permeability ;
            7. Roughness ;length
            8. Frequency ; frequency
            9. Physical Length ;length
            10. Impedance ;  impedance
            11. Electrical Length ; angle
            12. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            13. Conductor Loss ;   loss per length
            14. Dielectric Loss ;  loss per length
            15. Cutoff Frequency for TE11 mode ;  frequency

    """
    arg = arg[:9]
    newargs = convert2pq(arg, defaultunits)
    r, d, er, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    Z = Z_coaxial(er, r, d)
    eeff = er
    deg = electrical_length(eeff, freq, length)
    print("df ",eeff,freq,length,deg)
    cond_loss = conductor_loss_coaxial(er, r, d, freq, sigma, mu)
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    f_TE11 = (co/( pi*(r+d)*csqrt(mu*er)))
    argout = [Z, deg, eeff, cond_loss, diel_loss, f_TE11]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def coaxial_line_synthesis(arg, defaultunits):
    r"""Synthesis function for coaxial transmission line.

    Args:
        arg(list): First 9 arguments are inputs.

            1. Inner Radius (r);length
            2. Outer Radius (d);length
            3. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            4. Dielectric Loss Tangent ;
            5. Metal Conductivity ;  electrical conductivity
            6. Metal Permeability ;
            7. Roughness ;length
            8. Frequency ; frequency
            9. Physical Length ;length
            10. Impedance ;  impedance
            11. Electrical Length ; angle
            12. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            13. Conductor Loss ;   loss per length
            14. Dielectric Loss ;  loss per length
            15. Cutoff Frequency for TE11 mode ;  frequency

    Ref: Transmssion Line Design Handbook, p47, r-inner diameter, d-outer diameter
    """
    arg = arg[:11]
    newargs = convert2pq(arg, defaultunits)
    r, d, er, tand, sigma, mu, roughness, freq, length, Z, deg = tuple(newargs)
    output = Sentez(lambda *x: Z_coaxial(*x), [er, r, d], [1], target_value=[Z],
                    init_value=[(d/2.0)], limits=[((d/ 100.0), d * 100.0)])
    r = output[0]
    Z = Z_coaxial(er, r, d)
    eeff = er
    #deg = electrical_length(eeff, freq, length)
    length = physical_length(eeff, freq, deg)
    print("df ",eeff,freq,length,deg)
    cond_loss = conductor_loss_coaxial(er, r, d, freq, sigma, mu)
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    f_TE11 = (co/( pi*(r+d)*csqrt(mu*er)))
    argout = [eeff, cond_loss, diel_loss, f_TE11]
    arg[0] = prettystring(r, defaultunits[0])
    arg[8] = prettystring(length, defaultunits[8])
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def coaxial_analysis_view(arg, defaultunits):
    """
    """
    arg = arg[:9]
    newargs = convert2pq(arg, defaultunits)
    r, d, er, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    # visvis.clf()
    # gnd = visvis.solidCylinder(translation=(0,0,0),scaling=((d/r)+0.5,(d/r)+0.5,10.0),N=64)
    # diel = visvis.solidCylinder(translation=(0.0,0.0,-0.1),scaling=((d/r),(d/r),10.2),N=64)
    # line = visvis.solidCylinder(translation=(0.0,0.0,-0.2),scaling=((r/r),(r/r),10.4),N=64)
    # diel.faceColor="g"
    # line.faceColor="r"
    # gnd.faceColor="r"
    return

def coaxial_strip_center_analysis(arg, defaultunits):
    r"""Analysis function for coaxial transmission line with strip center conductor.

    Args:
        arg(list): First 9 arguments are inputs.

            1. Strip Width (w) ;length
            2. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            3. Outer Diameter (D);length
            4. Dielectric Loss Tangent ;
            5. Metal Conductivity ;  electrical conductivity
            6. Metal Permeability ;
            7. Roughness ;length
            8. Frequency ; frequency
            9. Physical Length ;length
            10. Impedance ;  impedance
            11. Electrical Length ; angle
            12. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            13. Conductor Loss ;   loss per length
            14. Dielectric Loss ;  loss per length
    Ref: Transmssion Line Design Handbook, p47, r-inner diameter, d-outer diameter
    """

    arg = arg[:9]
    newargs = convert2pq(arg, defaultunits)
    w, er, d, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    Z = Z_coaxial_strip_center(er, w, d)
    eeff = er
    deg = electrical_length(eeff, freq, length)
    sd = skin_depth(freq, sigma, mu)
    cond_loss = -mu * pi * freq / Z / co * (Z_coaxial_strip_center(1.0, w, d) - Z_coaxial_strip_center(
        1.0, w - sd, d + sd)) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    argout = [Z, deg, eeff, cond_loss, diel_loss]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def coaxial_strip_center_analysis_view(arg, defaultunits):
    """
    """
    arg = arg[:9]
    newargs = convert2pq(arg, defaultunits)
    w, er, d, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    # visvis.clf()
    # t=(w/5.0)
    # gnd = visvis.solidCylinder(translation=(-2.5*d/2/t,0,0),  scaling=(d/2/t+0.5,d/2/t+0.5,5.0*d/2/t),direction=(1,0,0),rotation=90,N=64)
    # diel = visvis.solidCylinder(translation=(-2.55*d/2/t,0,0),scaling=(d/2/t,d/2/t,5.1*d/2/t),        direction=(1,0,0),rotation=90,N=64)
    # line = visvis.solidBox(translation=(0,0,0),               scaling=((w/t),(t/t),5.2*d/2/t),            direction=(1,0,0),rotation=90)
    # diel.faceColor="g"
    # line.faceColor="r"
    # gnd.faceColor="r"
    return

def square_coaxial_circular_center_analysis(arg, defaultunits):
    r"""Analysis function for square coaxial transmission line with circular center conductor.

    Args:
        arg(list): First 9 arguments are inputs.

            1. Inner Radius (r);length
            2. Outer Size (D);length
            3. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            4. Dielectric Loss Tangent ;
            5. Metal Conductivity ;  electrical conductivity
            6. Metal Permeability ;
            7. Roughness ;length
            8. Frequency ; frequency
            9. Physical Length ;length
            10. Impedance ;  impedance
            11. Electrical Length ; angle
            12. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            13. Conductor Loss ;   loss per length
            14. Dielectric Loss ;  loss per length
            Ref: Transmssion Line Design Handbook, p47, r-inner diameter, d-outer diameter
    """

    arg = arg[:9]
    newargs = convert2pq(arg, defaultunits)
    r, d, er, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    Z = Z_square_coaxial(er, 2*r, d)
    eeff = er
    deg = electrical_length(eeff, freq, length)
    sd = skin_depth(freq, sigma, mu)
    cond_loss = -mu * pi * freq / Z / co * (Z_square_coaxial(1.0, 2*r, d) - Z_square_coaxial(
        1.0, 2*r - (sd/ 2.0), d + (sd/ 2.0))) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    argout = [Z, deg, eeff, cond_loss, diel_loss]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def rectangular_coaxial_line_analysis(arg, defaultunits):
    """Analysis function for rectangular coaxial transmission line.

    Args:
        arg(list): First 11 arguments are inputs.

            1. Line Width (w) ;length
            2. Line Thickness (t) ;length
            3. Box Width (a) ;length
            4. Box Height (b) ;length
            5. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            6. Dielectric Loss Tangent ;
            7. Metal Conductivity ;  electrical conductivity
            8. Metal Permeability ;
            9. Roughness ;length
            10. Frequency ; frequency
            11. Physical Length ;length
            12. Impedance ;  impedance
            13. Electrical Length ; angle
            14. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            15. Conductor Loss ;   loss per length
            16. Dielectric Loss ;  loss per length
            Ref: Transmssion Line Design Handbook, p60
    """

    arg = arg[:11]
    newargs = convert2pq(arg, defaultunits)
    w, t, a, b, er, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    Z = Z_rectangular_coaxial(w, b, t, a, er)
    eeff = er
    deg = electrical_length(eeff, freq, length)
    sd = skin_depth(freq, sigma, mu)
    cond_loss = -mu * pi * freq / Z / co * (Z_rectangular_coaxial(w, b, t, a, 1.0) - Z_rectangular_coaxial(
        w-sd, b + sd, t - sd, a+sd,1.0)) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    argout = [Z, deg, eeff, cond_loss, diel_loss]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def rectangular_coaxial_line_synthesis(arg, defaultunits):
    """Synthesis function for rectangular coaxial transmission line.

    Args:
        arg(list): First 11 arguments are inputs.

            1. Line Width (w) ;length
            2. Line Thickness (t) ;length
            3. Box Width (a) ;length
            4. Box Height (b) ;length
            5. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            6. Dielectric Loss Tangent ;
            7. Metal Conductivity ;  electrical conductivity
            8. Metal Permeability ;
            9. Roughness ;length
            10. Frequency ; frequency
            11. Physical Length ;length
            12. Impedance ;  impedance
            13. Electrical Length ; angle
            14. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            15. Conductor Loss ;   loss per length
            16. Dielectric Loss ;  loss per length
            Ref: Transmssion Line Design Handbook, p60
    """

    arg = arg[:13]
    newargs = convert2pq(arg, defaultunits)
    w, t, a, b, er, tand, sigma, mu, roughness, freq, length, Z ,deg = tuple(newargs)
    output = Sentez(lambda *x: Z_rectangular_coaxial(*x), [w, b, t, a, er], [0], target_value=[Z],
                    init_value=[b], limits=[((b/ 100.0), b * 100.0)])
    w = output[0]
    Z = Z_rectangular_coaxial(w, b, t, a, er)
    eeff = er
    length = physical_length(eeff, freq, deg)
    #deg = electrical_length(eeff, freq, length)
    sd = skin_depth(freq, sigma, mu)
    cond_loss = -mu * pi * freq / Z / co * (Z_rectangular_coaxial(w, b, t, a, 1.0) - Z_rectangular_coaxial(
        w-sd, b + sd, t - sd, a+sd,1.0)) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    argout = [eeff, cond_loss, diel_loss]
    arg[0] = prettystring(w, defaultunits[0])
    arg[10] = prettystring(length, defaultunits[10])
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def square_coaxial_line_square_center_analysis(arg, defaultunits):
    """Analysis function for square coaxial transmission line with square inner conductor.

    Args:
        arg(list): First 9 arguments are inputs.

            1. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            2. Inner Size (d) ;length
            3. Outer Size (D) ;length
            4. Dielectric Loss Tangent ;
            5. Metal Conductivity ;  electrical conductivity
            6. Metal Permeability ;
            7. Roughness ;length
            8. Frequency ; frequency
            9. Physical Length ;length
            10. Impedance ;  impedance
            11. Electrical Length ; angle
            12. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            13. Conductor Loss ;   loss per length
            14. Dielectric Loss ;  loss per length
            Ref: Transmssion Line Design Handbook, p47, r-inner diameter, d-outer diameter
    """

    arg = arg[:9]
    newargs = convert2pq(arg, defaultunits)
    er, r, d, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    Z = Z_square_coaxial_square_center(er, r, d)
    eeff = er
    deg = electrical_length(eeff, freq, length)
    sd = skin_depth(freq, sigma, mu)
    cond_loss = -mu * pi * freq / Z / co * (Z_square_coaxial_square_center(1.0, r, d) - Z_square_coaxial_square_center(
        1.0, r - (sd/ 2.0), d + (sd/ 2.0))) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    argout = [Z, deg, eeff, cond_loss, diel_loss]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def square_coaxial_line_square_center_synthesis(arg, defaultunits):
    """Synthesis function for square coaxial transmission line with square inner conductor.

    Args:
        arg(list): First 9 arguments are inputs.

            1. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            2. Inner Size ;length
            3. Outer Size ;length
            4. Dielectric Loss Tangent ;
            5. Metal Conductivity ;  electrical conductivity
            6. Metal Permeability ;
            7. Roughness ;length
            8. Frequency ; frequency
            9. Physical Length ;length
            10. Impedance ;  impedance
            11. Electrical Length ; angle
            12. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            13. Conductor Loss ;   loss per length
            14. Dielectric Loss ;  loss per length
            Ref: Transmssion Line Design Handbook, p47, r-inner diameter, d-outer diameter
    """

    arg = arg[:11]
    newargs = convert2pq(arg, defaultunits)
    er, r, d, tand, sigma, mu, roughness, freq, length, Z, deg = tuple(newargs)
    output = Sentez(lambda *x: Z_square_coaxial_square_center(*x), [er, r, d], [1], target_value=[Z],
                    init_value=[(d/2.0)], limits=[((d/ 1000.0), d * 1000.0)])
    r = output[0]
    Z = Z_square_coaxial_square_center(er, r, d)
    eeff = er
    length = physical_length(eeff, freq, deg)
    #deg = electrical_length(eeff, freq, length)
    sd = skin_depth(freq, sigma, mu)
    cond_loss = -mu * pi * freq / Z / co * (Z_square_coaxial_square_center(1.0, r, d) - Z_square_coaxial_square_center(
        1.0, r - (sd/ 2.0), d + (sd/ 2.0))) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    argout = [eeff, cond_loss, diel_loss]
    arg[1]=prettystring(r, defaultunits[1])
    arg[8]=prettystring(length, defaultunits[8])
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def eccentric_coaxial_analysis(arg, defaultunits):
    """Analysis function for eccentric coaxial transmission line.

    Args:
        arg(list): First 10 arguments are inputs.

            1. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>) ;
            2. Inner Radius (r) ;length
            3. Outer Radius (d) ;length
            4. Shift From Center (s) ;length
            5. Dielectric Loss Tangent ;
            6. Metal Conductivity ;  electrical conductivity
            7. Metal Permeability ;
            8. Roughness ;length
            9. Frequency ; frequency
            10. Physical Length ;length
            11. Impedance ;  impedance
            12. Electrical Length ; angle
            13. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            14. Conductor Loss ;   loss per length
            15. Dielectric Loss ;  loss per length
            Ref: Transmssion Line Design Handbook, p56, r-inner diameter, d-outer diameter
    """

    arg = arg[:10]
    newargs = convert2pq(arg, defaultunits)
    er, r, d, sh, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    Z = Z_eccentric_coaxial(er, r, d, sh)
    eeff = er
    deg = electrical_length(eeff, freq, length)
    sd = skin_depth(freq, sigma, mu)
    cond_loss = -mu * pi * freq / Z / co * (Z_eccentric_coaxial(1.0, r, d, sh) - Z_eccentric_coaxial(
        1.0, r - (sd/ 2.0), d + (sd/ 2.0), sh)) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    argout = [Z, deg, eeff, cond_loss, diel_loss]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def parallel_wires_analysis(arg, defaultunits):
    """Analysis function for parallel wires transmission line.

    Args:
        arg(list): First 10 arguments are inputs.

            1. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            2. First Diameter (<font size=+2>d<sub>1</sub></font>) ;length
            3. Second Diameter (<font size=+2>d<sub>2</sub></font>) ;length
            4. Center to Center Spacing (D) ;length
            5. Dielectric Loss Tangent ;
            6. Metal Conductivity ;  electrical conductivity
            7. Metal Permeability ;
            8. Roughness ;length
            9. Frequency ; frequency
            10. Physical Length ;length
            11. Impedance ;  impedance
            12. Electrical Length ; angle
            13. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            14. Conductor Loss ;   loss per length
            15. Dielectric Loss ;  loss per length
            Ref: Transmssion Line Design Handbook, p67
    """

    arg = arg[:10]
    newargs = convert2pq(arg, defaultunits)
    er, d1, d2, D, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    Z = Z_parallel_wires(er, d1, d2, D)
    eeff = er
    deg = electrical_length(eeff, freq, length)
    sd = skin_depth(freq, sigma, mu)
    cond_loss = -mu * pi * freq / Z / co * (Z_parallel_wires(1.0, d1, d2, D) - Z_parallel_wires(
        1.0, d1 - sd, d2 - sd, D)) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    argout = [Z, deg, eeff, cond_loss, diel_loss]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def Z_rectangular_coaxial(w, b, t, a, er):
    """
    Ref: Transmission line design handbook, p60
    """
    temp = w/b/(1.0-(t/b))+2.0/pi*log((1.0/(1.0-(t/b)))+(1.0/tanh(pi*a/2.0/b)))
    return eta0/4.0/sqrt(er)/temp

def Z_partial_coaxial(er, r, d):
    # My calculation, r-inner diameter, d1-inner diameter, d2-outer dielectric diameter, er1-inner dielectric eps, er2-outer dielectric eps
    # return eta0/2.0/pi*csqrt(log(d2/r)*(1./er1*log(d1/r)+1./er2*log(d2/r)))
    x = min(len(er), len(d))
    a = 0
    for i in range(x):
        a = a + 1. / er[i] * log((d[i]/ r))
    return eta0 / 2.0 / pi * csqrt(log((d[-1]/ r)) * a)

def er_eff_partial_coaxial(er, r, d):
    # My calculation, r-inner diameter, d1-inner diameter, d2-outer dielectric diameter, er1-inner dielectric eps, er2-outer dielectric eps
    # return eta0/2.0/pi*csqrt(log(d2/r)*(1./er1*log(d1/r)+1./er2*log(d2/r)))
    return (eta0 / 2.0 / pi * log((d[-1]/ r)) / Z_partial_coaxial(er, r, d)) ** 2.0

def conductor_loss_partial_coaxial(er, r, d, f):
    # Transmssion Line Design Handbook, p47, r-inner diameter, d-outer diameter
    pass

def Z_eeff_suspended_stripline_0(w, t, h, b, er, freq):
    # Model for Shielded Suspended Substrate Microstrip Line, Level 0
    # hu: ustteki hava boslugu
    # hl: alttaki hava boslugu
    # h: dielektrik kalinligi
    hu = hl = ((b-h)/2.0)
    hl1 = (0.4986-0.1397*log((h/hl)))**4
    h1 = (0.8621-0.1251*log((h/hl)))**4
    eeff=(1.0/(1.0+h/hl*(h1-hl1*log((w/hl)))*((1.0/csqrt(er))-1.0))**2)
    u = (w/(h+hl))
    fu = 6.0+(2.0*pi-6.0)*exp(-power((30.666/u),0.7528))
    Zo=60.0*log((fu/u)+csqrt(1.0+4/u/u))
    Z = (Zo/csqrt(eeff))
    return (Z, eeff)

def Z_eeff_inverted_suspended_stripline_0(w, t, h, b, er, freq):
    # Model for Shielded Suspended Substrate Microstrip Line.pdf, Level 0
    # hu: ustteki hava boslugu
    # hl: alttaki hava boslugu
    # h: dielektrik kalinligi
    hu = hl = ((b-h)/2.0)
    hl1 = (0.3092-0.1047*log((h/hl)))**2
    h1 = (0.5173-0.1515*log((h/hl)))**2
    eeff=(1.0+h/hl*(h1-hl1*log((w/hl)))*(csqrt(er)-1.0))**2
    u = (w/hl)
    fu = 6.0+(2.0*pi-6.0)*exp(-power((30.666/u),0.7528))
    Zo=60.0*log((fu/u)+csqrt(1.0+4/u/u))
    Z = (Zo/csqrt(eeff))
    return (Z, eeff)

def Z_eeff_suspended_microstripline(w, t, h, hl, er, freq):
    # Model for Shielded Suspended Substrate Microstrip Line.pdf, Level 1
    # hu: ustteki hava boslugu
    # hl: alttaki hava boslugu
    # h: dielektrik kalinligi
    # Over the range 0.5<=w/hl<=10, 0.05<=h/hl<=1.5, and er<=20 the accuracy
    # of these model equations (in reproducting the exact theoretical data) is generally
    # better than 0.6 percent.
    f1 = 1. - (1.0/ csqrt(er))
    f = log(er)
    d33 = (-35.2531 + 601.0291*f - 643.0814*f**2 + 161.2689*f**3) * 1e-9
    d32 = (62.855 - 3462.5*f + 2902.923*f**2 - 614.7068*f**3) * 1e-8
    d31 = (1956.3170 + 779.9975*f - 995.9494*f**2 + 183.1957*f**3) * 1e-6
    d30 = (-983.4028 - 255.1229*f + 455.8729*f**2 - 83.9468*f**3) * 1e-6
    d23 = (138.2037 - 1412.427*f + 1184.27*f**2 - 270.0047*f**3) * 1e-8
    d22 = (-532.1326 + 7274.721*f - 4955.738*f**2 + 941.4134*f**3) * 1e-7
    d21 = (-3931.09 - 1890.719*f + 1912.266*f**2 - 319.6794*f**3) * 1e-5
    d20 = (1954.072 + 333.3873*f - 700.7473*f**2 + 121.3212*f**3) * 1e-5
    d13 = (-1983.789 + 8523.929*f - 5235.46*f**2 + 1145.788*f**3) * 1e-8
    d12 = (999.3135 - 4036.791*f + 1762.412*f**2 - 298.0241*f**3) * 1e-6
    d11 = (2548.791 + 1531.931*f - 1027.5*f**2 + 138.4192*f**3) * 1e-4
    d10 = (-1410.205 + 149.9293*f + 198.2892*f**2 - 32.1679*f**3) * 1e-4
    d03 = (2481.569 + 1430.386*f + 10095.55*f**2 - 2599.132*f**3) * 1e-8
    d02 = (-3025.507 - 141.9368*f - 3099.47*f**2 - 777.6151*f**3) * 1e-6
    d01 = (4665.232 - 1790.4*f + 291.5858*f**2 - 8.0888*f**3) * 1e-4
    d00 = (176.2576 - 43.124*f + 13.4094*f**2 - 1.701*f**3) * 1e-2
    k = (hl/h)
    c3 = d33*k**3 + d32*k**2 + d31*k**1 + d30
    c2 = d23*k**3 + d22*k**2 + d21*k**1 + d20
    c1 = d13*k**3 + d12*k**2 + d11*k**1 + d10
    c0 = d03*k**3 + d02*k**2 + d01*k**1 + d00
    f2 = (1.0/ (c3*((w/hl))**3 + c2*((w/hl))**2 + c1*((w/hl))**1 + c0))
    print("f1, f2 = ",f1,f2)
    eeff = (1.0/(1.0-f1*f2)**2)
    u = (((w/hl))/(1.0+(h/hl)))
    fu = 6.0+(2.0*pi-6.0)*exp(-power((30.666/u),0.7528))
    Zo = 60.0*log((fu/u)+csqrt(1.0+4/u/u))
    Z = Zo/csqrt(eeff)
    return (Z, eeff)

def Z_eeff_inverted_suspended_stripline(w, t, h, hu, hl, er, freq):
    # Model for Shielded Suspended Substrate Microstrip Line.pdf, Level 1
    # hu: ustteki hava boslugu
    # hl: alttaki hava boslugu
    # h: dielektrik kalinligi
    # The stated error
    # of the fit to the exact theoretical calculations is less than 0.6% for 1<=er<=20,
    # 0.5<=w/hl<=10, and 0.06<=h/hl<=1.5.

    f = log(er)
    d33 = (- 530.2099 - 2666.352*f - 3220.096*f**2 + 1324.499*f**3) * 1e-9
    d32 = (124.9655  + 577.5381*f + 1366.453*f**2 - 481.13*f**3) * 1e-7
    d31 = (596.3251  + 188.1409*f- 1741.477*f**2 + 465.6756*f**3) * 1e-6
    d30 = (- 3170.21 - 1931.852*f  + 2715.327*f**2 - 519.342*f**3) * 1e-6
    d23 = (- 147.0235  + 62.4342*f  + 887.5211*f**2 - 270.7555*f**3) * 1e-7
    d22 = (253.893  + 158.5529*f - 3235.485*f**2 - 919.3661*f**3) * 1e-6
    d21 = (- 2823.481 - 1562.782*f + 3646.15*f**2 - 823.4223*f**3) * 1e-5
    d20 = (5602.767  + 4403.356*f - 4517.034*f**2 + 743.2717*f**3) * 1e-5
    d13 = (486.7425 + 279.8323*f - 431.3625*f**2 + 108.824*f**3) * 1e-6
    d12 = (- 1957.379 - 1170.936*f + 1480.857*f**2 - 347.6403*f**3) * 1e-5
    d11 = (915.5589  + 338.4033*f - 253.2933*f**2 + 40.4745*f**3) * 1e-3
    d10 = (219.066 - 253.0864*f  + 208.7469*f**2 - 27.3285*f**3) * 1e-3
    d03 = (- 556.0909 - 268.6165*f + 623.7094*f**2 - 119.1402*f**3) * 1e-6
    d02 = (1763.34 + 961.0481*f - 2089.28*f**2 + 375.8805*f**3) * 1e-5
    d01 = (4855.9472 - 3408.5207*f + 15296.73*f**2 - 2418.1785*f**3) * 1e-5
    d00 = (2359.401 - 97.1644*f - 5.7706*f**2 + 11.4112*f**3) * 1e-3
    k = (hl/h)
    c3 = d33*k**3 + d32*k**2 + d31*k**1 + d30
    c2 = d23*k**3 + d22*k**2 + d21*k**1 + d20
    c1 = d13*k**3 + d12*k**2 + d11*k**1 + d10
    c0 = d03*k**3 + d02*k**2 + d01*k**1 + d00
    f2 = (1./ (c3*((w/hl))**3 + c2*((w/hl))**2 + c1*((w/hl))**1 + c0))
    f1 = csqrt(er)-1.0
    eeff = (1.0+f1*f2)**2
    b = h+hu+hl
    u = (w/b)
    fu = 6.0+(2.0*pi-6.0)*exp(-power((30.666/u),0.7528))
    Zo=60.0*log((fu/u)+csqrt(1.0+4/u/u))
    Z = (Zo/csqrt(eeff))
    return (Z, eeff)

def Z_eeff_suspended_stripline_eski(w, t, a, b, er, freq):
    # Transmssion Line Design Handbook, p141, a-dielectric height, b-spacing
    # height, t-metal thickness, w-metal width
    # Hatali, er etkisi olmasi gerekenden az gorunuyor.
    f1 = 1. - (1.0/ csqrt(er))
    f = log(er)
    d33 = (-35.2531 + 601.0291 * f - 643.0814 * f **
           2 + 161.2689 * f ** 3) * 1e-9
    d32 = (62.855 - 3462.5 * f + 2902.923 * f ** 2 - 614.7068 * f ** 3) * 1e-8
    d31 = (1956.3170 + 779.9975 * f - 995.9494 * f **
           2 + 183.1957 * f ** 3) * 1e-6
    d30 = (-983.4028 - 255.1229 * f + 455.8729 * f **
           2 - 83.9468 * f ** 3) * 1e-6
    d23 = (138.2037 - 1412.427 * f + 1184.27 * f **
           2 - 270.0047 * f ** 3) * 1e-8
    d22 = (-532.1326 + 7274.721 * f - 4955.738 * f **
           2 + 941.4134 * f ** 3) * 1e-7
    d21 = (-3931.09 - 1890.719 * f + 1912.266 * f **
           2 - 319.6794 * f ** 3) * 1e-5
    d20 = (1954.072 + 333.3873 * f - 700.7473 * f **
           2 + 121.3212 * f ** 3) * 1e-5
    d13 = (-1983.789 + 8523.929 * f - 5235.46 * f **
           2 + 1145.788 * f ** 3) * 1e-8
    d12 = (999.3135 - 4036.791 * f + 1762.412 * f **
           2 - 298.0241 * f ** 3) * 1e-6
    d11 = (2548.791 + 1531.931 * f - 1027.5 * f **
           2 + 138.4192 * f ** 3) * 1e-4
    d10 = (-1410.205 + 149.9293 * f + 198.2892 * f **
           2 - 32.1679 * f ** 3) * 1e-4
    d03 = (2481.569 + 1430.386 * f + 10095.55 * f **
           2 - 2599.132 * f ** 3) * 1e-8
    d02 = (-3025.507 - 141.9368 * f - 3099.47 * f **
           2 + 777.6151 * f ** 3) * 1e-6
    d01 = (4665.232 - 1790.4 * f + 291.5858 * f ** 2 - 8.0888 * f ** 3) * 1e-4
    d00 = (176.2576 - 43.124 * f + 13.4094 * f ** 2 - 1.701 * f ** 3) * 1e-2
    c3 = (1./ (d33 * ((b/ a)) ** 3 + d32 * ((b/ a))
               ** 2 + d31 * ((b/ a)) ** 1 + d30))
    c2 = (1./ (d23 * ((b/ a)) ** 3 + d22 * ((b/ a))
               ** 2 + d21 * ((b/ a)) ** 1 + d20))
    c1 = (1./ (d13 * ((b/ a)) ** 3 + d12 * ((b/ a))
               ** 2 + d11 * ((b/ a)) ** 1 + d10))
    c0 = (1./ (d03 * ((b/ a)) ** 3 + d02 * ((b/ a))
               ** 2 + d01 * ((b/ a)) ** 1 + d00))
    f2 = (1./ (c3 * ((w/ b)) ** 3 + c2 * ((w/ b)) ** 2 + c1 * ((w/ b)) ** 1 + c0))
    u = (((w/ b))/ (1 + (a/ b)))
    fu = 6. + (2 * pi - 6.) * exp(-((30.666/ u)) ** (0.7528))
    eeff = (1./ (1. - f1 * f2) ** 2)
    Z = eta0 / (2. * pi) * log((fu/ u) + csqrt(1. + (4.0/ u ** 2)))
    Zo = (Z/ csqrt(eeff))

    C2 = 0.0004 * ((a/ b)) ** 2 - 0.0004 * ((a/ b)) ** 3 + 0.0001 * ((a/ b)) ** 4
    C1 = -0.0008 + 0.0096 * \
        ((a/ b)) - 0.0346 * ((a/ b)) ** 2 + 0.0384 * \
        ((a/ b)) ** 3 - 0.0135 * ((a/ b)) ** 4
    C0 = 0.0194 - 0.2398 * \
        ((a/ b)) + 0.8977 * ((a/ b)) ** 2 - 0.9924 * \
        ((a/ b)) ** 3 + 0.3468 * ((a/ b)) ** 4
    G = C0 + C1 * Z + C2 * Z ** 2
    fp = Zo * a * 100 / (2.0 * pi * mu0 * 0.064 * (a + b))
    eeff = er - ((er - eeff)/ (1 + G * ((freq/ fp)) ** 2))
    Zo = (Z/ csqrt(eeff))
    return (Zo, eeff)

def covered_suspended_microstripline_analysis(arg, defaultunits):
    r"""Analysis function for the covered suspended microstrip transmission line.

    Args:
        arg(list): First 12 arguments are inputs.

            1. Line Width (w) ;length
            2. Metal Thickness (t) ;length
            3. Substrate Thickness (h) ;length
            4. Upper Cavity Height (hu) ;length
            5. Lower Cavity Height (hl) ;length
            6. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            7. Dielectric Loss Tangent ;
            8. Metal Conductivity ;  electrical conductivity
            9. Metal Permeability ;
            10. Roughness ;length
            11. Frequency ; frequency
            12. Physical Length ;length
            13. Impedance ;   impedance
            14. Electrical Length ;  angle
            15. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            16. Conductor Loss ;   loss per length
            17. Dielectric Loss ;   loss per length
            Ref: Model for Shielded Suspended Substrate Microstrip Line.pdf, Level 1
            Over the range 0.5<=w/hl<=10, 0.05<=h/hl<=1.5, and er<=20 the accuracy
            of these model equations (in reproducing the exact theoretical data) is generally
            better than 0.6 percent.
            Static Model. Does not use frequency.
            Does not use thickness.
    """

    # Ref: Transmssion Line Design Handbook, p141, a-dielectric height, b-spacing height, t-metal thickness, w-metal width
    # Dispersion characteristics are valid for er=12.9 and frequency >20GHz

    arg = arg[:12]
    newargs = convert2pq(arg, defaultunits)
    w, t, h, hu, hl, er, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    Z, eeff = Z_eeff_suspended_stripline(w, t, h, hu, hl, er, freq)
    deg = electrical_length(eeff, freq, length)
    sd = min([skin_depth(freq, sigma, mu),(t/2.0)])
    cond_loss = -mu * pi * freq / Z / co * (Z_eeff_suspended_stripline(w, t, h, hu, hl, 1.0, freq)[0] - Z_eeff_suspended_stripline(
        w - sd, t - sd,  h, hu + sd, hl + sd, 1.0, freq)[0]) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    argout = [Z, deg, eeff, cond_loss, diel_loss]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def covered_suspended_microstripline_synthesis(arg, defaultunits):
    """Synthesis function for the covered suspended microstrip transmission line.

    Args:
        arg(list): First 11 arguments are inputs.

            1. Line Width (w) ;length
            2. Metal Thickness (t) ;length
            3. Substrate Thickness (h) ;length
            4. Upper Cavity Heigh (hu) ;length
            5. Lower Cavity Height (hl) ;length
            6. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            7. Dielectric Loss Tangent ;
            8. Metal Conductivity ;  electrical conductivity
            9. Metal Permeability ;
            10. Roughness ;length
            11. Frequency ; frequency
            12. Physical Length ;length
            13. Impedance ;   impedance
            14. Electrical Length ;  angle
            15. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            16. Conductor Loss ;   loss per length
            17. Dielectric Loss ;   loss per length
            Ref: Transmssion Line Design Handbook, p141, a-dielectric height, b-spacing height, t-metal thickness, w-metal width
            Dispersion characteristics are valid for er=12.9 and frequency >20GHz
    """

    arg = arg[:13]
    newargs = convert2pq(arg, defaultunits)
    w, t, h, hu, hl, er, tand, sigma, mu, roughness, freq, length, Z, deg = tuple(newargs)
    # output =Sentez(lambda *x:Z_eeff_suspended_stripline(*x)[0], [w, t, a, b, er, freq], [0],target_value=[Z],init_value=[b], limits = [(a/100.0,a*100.0),(a/100.0,a*100.0)])
    output =Sentez(lambda *x:Z_eeff_suspended_stripline(*x)[0], [w, t, h, hu, hl, er, freq], [0],target_value=[Z],init_value=[b], limits = [((a/1000.0),a*1000.0)])
    w= output[0]
    Z, eeff = Z_eeff_suspended_stripline(w, t, h, hu, hl, er, freq)
    length = physical_length(eeff, freq, deg)
    sd = skin_depth(freq, sigma, mu)
    cond_loss = -mu * pi * freq / Z / co * (Z_eeff_suspended_stripline(w, t, h, hu, hl, 1.0, freq)[0] - Z_eeff_suspended_stripline(
        w - sd, t - sd,  h, hu + sd, hl + sd, 1.0, freq)[0]) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    argout = [Z, deg, eeff, cond_loss, diel_loss]
    arg[0]=prettystring(w, defaultunits[0])
    arg[10]=prettystring(length, defaultunits[10])
    arg = arg[:11]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def suspended_microstripline_analysis(arg, defaultunits):
    """Analysis function for the suspended microstrip transmission line.

    Args:
        arg(list): First 11 arguments are inputs.

            1. Line Width (w) ;length
            2. Metal Thickness (t) ;length
            3. Substrate Thickness (a) ;length
            4. Spacing Height (b) ;length
            5. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            6. Dielectric Loss Tangent ;
            7. Metal Conductivity ;  electrical conductivity
            8. Metal Permeability ;
            9. Roughness ;length
            10. Frequency ; frequency
            11. Physical Length ;length
            12. Impedance ;   impedance
            13. Electrical Length ;  angle
            14. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            15. Conductor Loss ;   loss per length
            16. Dielectric Loss ;   loss per length
            Ref: Model for Shielded Suspended Substrate Microstrip Line.pdf, Level 1
            Over the range 0.5<=w/hl<=10, 0.05<=h/hl<=1.5, and er<=20 the accuracy
            of these model equations (in reproducing the exact theoretical data) is generally
            better than 0.6 percent.
            Static Model. Does not use frequency.
            Does not use thickness.
    """

    # Ref: Transmssion Line Design Handbook, p141, a-dielectric height, b-spacing height, t-metal thickness, w-metal width
    # Dispersion characteristics are valid for er=12.9 and frequency >20GHz

    arg = arg[:11]
    newargs = convert2pq(arg, defaultunits)
    w, t, a, b, er, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    Z, eeff = Z_eeff_suspended_stripline(w, t, a, 1000*b, b, er, freq)
    deg = electrical_length(eeff, freq, length)
    sd = min(skin_depth(freq, sigma, mu),(t/2.0))
    cond_loss = -mu * pi * freq / Z / co * (Z_eeff_suspended_stripline(w, t, a, 1000*b, b, 1.0, freq)[0] - Z_eeff_suspended_stripline(
        w - sd, t - sd, a, 1000*b, b + sd, 1.0, freq)[0]) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    argout = [Z, deg, eeff, cond_loss, diel_loss]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def suspended_microstripline_synthesis(arg, defaultunits):
    """Synthesis function for the suspended microstrip transmission line.

    Args:
        arg(list): First 11 arguments are inputs.

            1. Line Width (w) ;length
            2. Metal Thickness (t) ;length
            3. Substrate Thickness (a) ;length
            4. Spacing Height (b) ;length
            5. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            6. Dielectric Loss Tangent ;
            7. Metal Conductivity ;  electrical conductivity
            8. Metal Permeability ;
            9. Roughness ;length
            10. Frequency ; frequency
            11. Physical Length ;length
            12. Impedance ;   impedance
            13. Electrical Length ;  angle
            14. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            15. Conductor Loss ;   loss per length
            16. Dielectric Loss ;   loss per length
            Ref: Transmssion Line Design Handbook, p141, a-dielectric height, b-spacing height, t-metal thickness, w-metal width
            Dispersion characteristics are valid for er=12.9 and frequency >20GHz
    """

    arg = arg[:13]
    newargs = convert2pq(arg, defaultunits)
    w, t, a, b, er, tand, sigma, mu, roughness, freq, length, Z, deg = tuple(newargs)
    # output =Sentez(lambda *x:Z_eeff_suspended_stripline(*x)[0], [w, t, a, b, er, freq], [0],target_value=[Z],init_value=[b], limits = [(a/100.0,a*100.0),(a/100.0,a*100.0)])
    output =Sentez(lambda *x:Z_eeff_suspended_microstripline(*x)[0], [w, t, a, b, er, freq], [0],target_value=[Z],init_value=[b], limits = [((a/1000.0),a*1000.0)])
    w= output[0]


    Z, eeff = Z_eeff_suspended_microstripline(w, t, a, b, er, freq)
    length = physical_length(eeff, freq, deg)
    sd = skin_depth(freq, sigma, mu)
    cond_loss = -mu * pi * freq / Z / co * (Z_eeff_suspended_microstripline(w, t, a, b, 1.0, freq)[0] - Z_eeff_suspended_microstripline(
        w - sd, t - sd, a, b + sd, 1.0, freq)[0]) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    argout = [Z, deg, eeff, cond_loss, diel_loss]
    arg[0]=prettystring(w, defaultunits[0])
    arg[10]=prettystring(length, defaultunits[10])
    arg = arg[:11]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def Z_eeff_shielded_suspended_stripline(w, h, b, a, er):
    # Transmssion Line Design Handbook, p145, h-dielectric height, b-total height, a-box width, w-metal width
    # Validity: 1<a/b<2.5, 1<er<4, 0.1<h/b<0.5
    if w <= ((a/ 2)):
        f = 0.03451 - 0.1031 * h / b + 0.01742 * a / b
        e = 0.2077 + 1.2177 * h / b - 0.08364 * a / b
        r = 1.0835 + 0.1007 * h / b - 0.09457 * a / b
        # Bu formul kitapta hatali
        v = -1.7866 - 0.2035 * h / b + 0.475 * a / b
        eeff = (1.0/ \
            (1.0 + (e - f * log((w/ b))) * log((1.0/ csqrt(er)))) ** 2)
        Zo = eta0 / 2 / pi / \
            csqrt(eeff) * \
            (v + r * log(6.0 / w * b + csqrt(1.0 + (4.0/ ((w/ b)) ** 2))))
    else:
        f = -0.1424 + 0.3017 * h / b - 0.02411 * a / b
        e = 0.464 + 0.9647 * h / b - 0.2063 * a / b
        r = 1.9492 + 0.1553 * h / b - 0.5123 * a / b
        v = -0.6301 - 0.07082 * h / b + 0.247 * a / b
        eeff = (1.0/ \
            (1.0 + (e - f * log((w/ b))) * log((1.0/ csqrt(er)))) ** 2)
        Zo = eta0 / \
            csqrt(eeff) * \
            (v + (r/ ((w/ b) + 1.393 + 0.667 * log((w/ b) + 1.444))))
    return (Zo, eeff)

def shielded_suspended_stripline_analysis(arg, defaultunits):
    r"""Analysis function for the shielded suspended stripline transmission line.

    Args:
        arg(list): First 11 arguments are inputs.

            1. Line Width (w);length
            2. Substrate Thickness (h);length
            3. Total Height (b);length
            4. Box Width (a);length
            5. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            6. Dielectric Loss Tangent ;
            7. Metal Conductivity ;  electrical conductivity
            8. Metal Permeability ;
            9. Roughness ;length
            10. Frequency ; frequency
            11. Physical Length ;length
            12. Impedance ;   impedance
            13. Electrical Length ;   angle
            14. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            15. Conductor Loss ;   loss per length
            16. Dielectric Loss ;  loss per length
            Ref: Transmssion Line Design Handbook, p141
            Analysis Equations for Shielded Suspended Substrate Microstrip Line and Broadside-Coupled Stripline.pdf
            Valid for 1 < a/b < 2.5, 1 < er < 4, 0.1 < h/b < 0.5
    """
    conditions = ["1.0 < a/b < 2.5","1.0 < er < 4.0", "0.1 < h/b < 0.5"]
    self = globals()[sys._getframe().f_code.co_name]

    arg = arg[:11]
    newargs = convert2pq(arg, defaultunits)
    w, h, b, a, er, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    Z, eeff = Z_eeff_shielded_suspended_stripline(w, h, b, a, er)
    deg = electrical_length(eeff, freq, length)
    sd = skin_depth(freq, sigma, mu)
    cond_loss = -mu * pi * freq / Z / co * (Z_eeff_shielded_suspended_stripline(w, h, b, a, 1.0)[0] - Z_eeff_shielded_suspended_stripline(
        w - sd, h + sd, b + sd, a + sd, 1.0)[0]) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    argout = [Z, deg, eeff, cond_loss, diel_loss]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    self.warnings = [x for x in conditions if not eval(x)]
    return arg

def shielded_suspended_stripline_synthesis(arg, defaultunits):
    """Synthesis function for the shielded suspended stripline transmission line.

    Args:
        arg(list): First 11 arguments are inputs.

            1. Line Width (w);length
            2. Substrate Thickness (h);length
            3. Total Height (b);length
            4. Box Width (a);length
            5. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            6. Dielectric Loss Tangent ;
            7. Metal Conductivity ;  electrical conductivity
            8. Metal Permeability ;
            9. Roughness ;length
            10. Frequency ; frequency
            11. Physical Length ;length
            12. Impedance ;   impedance
            13. Electrical Length ;   angle
            14. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            15. Conductor Loss ;   loss per length
            16. Dielectric Loss ;  loss per length
            Ref: Transmssion Line Design Handbook, p141
            Analysis Equations for Shielded Suspended Substrate Microstrip Line and Broadside-Coupled Stripline.pdf
            Valid for 1 < a/b < 2.5, 1 < er < 4, 0.1 < h/b < 0.5
    """

    arg = arg[:13]
    newargs = convert2pq(arg, defaultunits)
    w, h, b, a, er, tand, sigma, mu, roughness, freq, length,Z,deg = tuple(newargs)
    # Z, eeff = Z_eeff_shielded_suspended_stripline(w, h, b, a, er)
    output =Sentez(lambda *x:Z_eeff_shielded_suspended_stripline(*x)[0], [w, h, b, a, er], [0],target_value=[Z],init_value=[b], limits = [((a/100.0),a*100.0)])
    w= output[0]
    Z, eeff = Z_eeff_shielded_suspended_stripline(w, h, b, a, er)
    length = physical_length(eeff, freq, deg)
    sd = skin_depth(freq, sigma, mu)
    cond_loss = -mu * pi * freq / Z / co * (Z_eeff_shielded_suspended_stripline(w, h, b, a, 1.0)[0] - Z_eeff_shielded_suspended_stripline(
        w - sd, h + sd, b + sd, a + sd, 1.0)[0]) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    argout = [eeff, cond_loss, diel_loss]
    arg[0]=prettystring(w, defaultunits[0])
    arg[10]=prettystring(length, defaultunits[10])
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def Z_eeff_grounded_cpw(w, er, s, h):
    """Coplanar waveguide circuits, components and systems s89
    Transmission Line Design Handbook s79
    """
    a = (w/ 2)
    b = (w/ 2) + s
    k = (a/ b)
    k3 = (tanh(pi * a / 2 / h)/ tanh(pi * b / 2 / h))
    k_ = csqrt(1.0 - k ** 2)
    k3_ = csqrt(1.0 - k3 ** 2)
    x = ellipk(k_) * ellipk(k3) / (ellipk(k) * ellipk(k3_))
    eeff = ((1 + er * x)/ (1 + x))
    Zo = 60 * pi /csqrt(eeff) / ((ellipk(k)/ ellipk(k_)) + (ellipk(k3)/ ellipk(k3_)))
    return (Zo, eeff)

def Z_eeff_grounded_cpw_thick(w, th, er, s, h):
    """Coplanar waveguide circuits, components and systems s89
    Transmission Line Design Handbook s79
	For thickness correction Reference: "CPWG impedance formula" document
    """
    dd = 1.25*th/pi*(1.0+log(2*h/th));
    Zair, _ = Z_eeff_grounded_cpw(w+dd, 1.0, s-dd, h)
    Cair = 1/co/Zair
    L = Cair*Zair*Zair
    Zair_thin, _ = Z_eeff_grounded_cpw(w, 1.0, s, h)
    Cex = 1./co*(1./Zair-1./Zair_thin)
    Cex = 2.0*eps0*th/s+(Cex-2*eps0*th/s)*(er+1.0)/2.0
    Zthin, eps_eff_thin = Z_eeff_grounded_cpw(w, er, s, h)
    Cthin = 1./Zthin/co*sqrt(eps_eff_thin);
    vp = 1./sqrt(L*(Cthin+Cex));
    eeff = (co/vp)**2;
    Zo = sqrt(L/(Cthin+Cex));
    return (Zo, eeff)

def Z_eeff_cpw(w, er, s, h, t):
    """ Transmission Line Design Handbook s73"""
    a = w
    b = w + 2 * s
    k = (a/ b)
    at = a+1.25*t/pi*(1.0+log(4.0*pi*a/t))
    bt = b-1.25*t/pi*(1.0+log(4.0*pi*a/t))
    kt = (at/bt)
    k_=csqrt(1.0-k*k)
    kt_=csqrt(1.0-kt*kt)
    k1=(sinh(pi*at/4.0/h)/ sinh(pi*bt/4.0/h))
    k1_=csqrt(1.0-k1*k1)
    eeff=1.0+(er-1.0)/2.0*ellipk(k_) * ellipk(k1) / (ellipk(k) * ellipk(k1_))
    eeff_t=eeff-((eeff-1.0)/((b-a)/2.0/0.7/t*ellipk(k)/ellipk(k_)+1.0))
    Zo=30.0*pi*ellipk(kt_)/ellipk(kt)/csqrt(eeff_t)
    return (Zo, eeff_t)

def grounded_cpw_analysis(arg, defaultunits):
    r"""Analysis function for the grounded coplanar waveguide transmission line.

    Args:
        arg(list): First 11 arguments are inputs.

            1. Line Width (w);length
            2. Line Spacing (s);length
            3. Metal Thickness (th);length
            4. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            5. Substrate Thickness (h);length
            6. Dielectric Loss Tangent ;
            7. Metal Conductivity ;  electrical conductivity
            8. Metal Permeability ;
            9. Roughness ;length
            10. Frequency ; frequency
            11. Physical Length ;length
            12. Impedance ;   impedance
            13. Electrical Length ; angle
            14. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            15. Conductor Loss ;   loss per length
            16. Dielectric Loss ;  loss per length
            Ref: Coplanar waveguide circuits, components and systems s89
    """

    arg = arg[:10]
    newargs = convert2pq(arg, defaultunits)
    w, s, th, er, h, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    Z, eeff = Z_eeff_grounded_cpw_thick(w, th, er, s, h)
    deg = electrical_length(eeff, freq, length)
    sd = skin_depth(freq, sigma, mu)
    cond_loss = -mu * pi * freq / Z / co * (Z_eeff_grounded_cpw(w, th, 1.0, s, h)[0] - Z_eeff_grounded_cpw(
        w - sd, th-sd, 1.0, s + sd, h + sd)[0]) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    argout = [Z, deg, eeff, cond_loss, diel_loss]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i]) for i in range(len(argout))]
    return arg

def grounded_cpw_synthesis(arg, defaultunits):
    r"""Synthesis function for the grounded coplanar waveguide transmission line.

    Args:
        arg(list): First 10 arguments are inputs.

            1. Line Width (w);length
            2. Line Spacing (s);length
            3. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            4. Substrate Thickness (h);length
            5. Dielectric Loss Tangent ;
            6. Metal Conductivity ;  electrical conductivity
            7. Metal Permeability ;
            8. Roughness ;length
            9. Frequency ; frequency
            10. Physical Length ;length
            11. Impedance ;   impedance
            12. Electrical Length ; angle
            13. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            14. Conductor Loss ;   loss per length
            15. Dielectric Loss ;  loss per length
            Ref: Coplanar waveguide circuits, components and systems s89
    """

    arg = arg[:12]
    newargs = convert2pq(arg, defaultunits)
    w, s, er, h, tand, sigma, mu, roughness, freq, length, Z, deg = tuple(newargs)
    output = Sentez(lambda *x: Z_eeff_grounded_cpw(*x)[0], [w, er, s, h], [0], target_value=[Z],
                    init_value=[h], limits=[((h/ 1000.0), h * 1000.0)])
    w = output[0]
    Z, eeff = Z_eeff_grounded_cpw(w, er, s, h)
    length = physical_length(eeff, freq, deg)
    sd = skin_depth(freq, sigma, mu)
    cond_loss = -mu * pi * freq / Z / co * (Z_eeff_grounded_cpw(w, 1.0, s, h)[0] - Z_eeff_grounded_cpw(
        w - sd, 1.0, s + sd, h + sd)[0]) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    arg[0] = prettystring(w, defaultunits[0])
    arg[9] = prettystring(length, defaultunits[9])
    arg = arg[:10]
    argout = [Z, deg, eeff, cond_loss, diel_loss]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i]) for i in range(len(argout))]
    return arg

def Z_eeff_covered_grounded_cpw(w, s, h, er, h1):
    """ Coplanar waveguide circuits, components and systems s89"""
    a = (w/ 2)
    b = (w/ 2) + s
    k = (a/ b)
    k3 = (tanh(pi * a / 2 / h)/ tanh(pi * b / 2 / h))
    k4 = (tanh(pi * a / 2 / h1)/ tanh(pi * b / 2 / h1))
    k4_ = sqrt(1.0 - k4 ** 2)
    k3_ = sqrt(1.0 - k3 ** 2)
    x = ellipk(k4) / ellipk(k4_) / ((ellipk(k3)/ ellipk(k3_)))
    q = (1./ (1. + x))
    eeff = 1. + q * (er - 1.)
    Zo = 60 * pi / \
        csqrt(eeff) / ((ellipk(k3)/ ellipk(k3_)) + (ellipk(k4)/ ellipk(k4_)))
    return (Zo, eeff)

def covered_grounded_coplanar_waveguide_analysis(arg, defaultunits):
    r"""Analysis function for the covered grounded coplanar waveguide transmission line.

    Args:
        arg(list): First 11 arguments are inputs.

            1. Line Width (w);length
            2. Line Spacing (s);length
            3. Substrate Thickness (h);length
            4. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            5. Cover Height (b);length
            6. Dielectric Loss Tangent ;
            7. Metal Conductivity ;  electrical conductivity
            8. Metal Permeability ;
            9. Roughness ;length
            10. Frequency ; frequency
            11. Physical Length ;length
            11. Impedance ;   impedance
            12. Electrical Length ;  angle
            13. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            14. Conductor Loss ; loss per length
            15. Dielectric Loss ;   loss per length
            Ref: Coplanar waveguide circuits, components and systems s89
    """

    arg = arg[:11]
    newargs = convert2pq(arg, defaultunits)
    w, s, h, er, h1, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    Z, eeff = Z_eeff_covered_grounded_cpw(w, s, h, er, h1)
    deg = electrical_length(eeff, freq, length)
    sd = skin_depth(freq, sigma, mu)
    cond_loss = -mu * pi * freq / Z / co * (Z_eeff_covered_grounded_cpw(w, s, h, 1.0, h1)[0] - Z_eeff_covered_grounded_cpw(
        w - sd, s + sd, h + sd, 1.0, h1 + sd)[0]) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    argout = [Z, deg, eeff, cond_loss, diel_loss]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i]) for i in range(len(argout))]
    return arg

def covered_grounded_coplanar_waveguide_synthesis(arg, defaultunits):
    r"""Synthesis function for the covered grounded coplanar waveguide transmission line.

    Args:
        arg(list): First 11 arguments are inputs.

            1. Line Width (w);length
            2. Line Spacing (s);length
            3. Substrate Thickness (h);length
            4. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            5. Cover Height (b);length
            6. Dielectric Loss Tangent ;
            7. Metal Conductivity ;  electrical conductivity
            8. Metal Permeability ;
            9. Roughness ;length
            10. Frequency ; frequency
            11. Physical Length ;length
            12. Impedance ;   impedance
            13. Electrical Length ;  angle
            14. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            15. Conductor Loss ; loss per length
            16. Dielectric Loss ;   loss per length
            Ref: Coplanar waveguide circuits, components and systems s89
    """

    arg = arg[:13]
    newargs = convert2pq(arg, defaultunits)
    w, s, h, er, h1, tand, sigma, mu, roughness, freq, length, Z, deg = tuple(newargs)
    output = Sentez(lambda *x: Z_eeff_covered_grounded_cpw(*x)[0], [w, s, h, er, h1], [0], target_value=[Z],
                    init_value=[h], limits=[((h/ 100.0), h * 100.0)])
    w = output[0]
    Z, eeff = Z_eeff_covered_grounded_cpw(w, s, h, er, h1)
    length = physical_length(eeff, freq, deg)
    sd = skin_depth(freq, sigma, mu)
    cond_loss = -mu * pi * freq / Z / co * (Z_eeff_covered_grounded_cpw(w, s, h, 1.0, h1)[0] - Z_eeff_covered_grounded_cpw(
        w - sd, s + sd, h + sd, 1.0, h1 + sd)[0]) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    argout = [Z, deg, eeff, cond_loss, diel_loss]
    arg[0]=prettystring(w, defaultunits[0])
    arg[10]=prettystring(length, defaultunits[10])
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i]) for i in range(len(argout))]
    return arg

def Z_eeff_laterally_covered_grounded_cpw(w, s, h, er, h1):
    """Coplanar waveguide circuits, components and systems s89"""
    pass

def edge_coupled_microstrip_analysis(arg, defaultunits):
    r"""Analysis function for the edge coupled microstrip transmission line.

    Args:
        arg(list): First 11 arguments are inputs.

            1. Line Width (w) ;length
            2. Line Gap (s) ;length
            3. Metal Thickness (t) ;length
            4. Substrate Thickness (h) ;length
            5. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            6. Dielectric Loss Tangent ;
            7. Metal Conductivity ;  electrical conductivity
            8. Metal Permeability ;
            9. Roughness ;length
            10. Frequency ; frequency
            11. Physical Length ;length
            12. Impedance (even);   impedance
            13. Impedance (odd);   impedance
            14. Electrical Length (even) ;   angle
            15. Electrical Length (odd) ;  angle
            16. <font size=+2>&epsilon;<sub>eff</sub></font> (even);
            17. <font size=+2>&epsilon;<sub>eff</sub></font> (odd);
            18. Conductor Loss (even) ;  loss per length
            19. Conductor Loss (odd) ;  loss per length
            20. Dielectric Loss (even) ; loss per length
            21. Dielectric Loss (odd) ; loss per length
            22. Maximum Coupling ;
            23. Matched Impedance ;
            Ref: Transmssion Line Design Handbook, p199, with errata sheet
    """

    arg = arg[:11]
    newargs = convert2pq(arg, defaultunits)
    w, s, t, h, er, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    Z_even, Z_odd, eeff_even, eeff_odd = Z_eeff_edge_coupled_microstrip(
        w, er, t, h, s, freq)
    deg_even = electrical_length(eeff_even, freq, length)
    deg_odd = electrical_length(eeff_odd, freq, length)
    sd = skin_depth(freq, sigma, mu)
    # print "sd ",sd
    # print "ze ",Z_even
    # print "w-sd ",w-sd
    # print "t-sd ",t-sd
    Z_even1, Z_odd1, eeff_even1, eeff_odd1 = Z_eeff_edge_coupled_microstrip(
        w - sd, 1.001, t - sd, h + sd, s + sd, freq)
    Z_even2, Z_odd2, eeff_even2, eeff_odd2 = Z_eeff_edge_coupled_microstrip(
        w, 1.001, t, h, s, freq)
    cond_loss_even = -pi * freq / Z_even / co * \
        (Z_even2 - Z_even1) * 20.0 * \
        log10(exp(1))  # dB/m, incremental inductance
    cond_loss_odd = -pi * freq / Z_odd / co * \
        (Z_odd2 - Z_odd1) * 20.0 * \
        log10(exp(1))  # dB/m, incremental inductance
    diel_loss_even = dielectric_loss(eeff_even, er, freq, tand)
    diel_loss_odd = dielectric_loss(eeff_odd, er, freq, tand)
    max_coupling = 20 * log10(((Z_even - Z_odd)/ (Z_even + Z_odd)))
    matched_impedance = sqrt((Z_even * Z_odd))
    argout = [Z_even, Z_odd, deg_even, deg_odd, eeff_even, eeff_odd,
              cond_loss_even, cond_loss_odd, diel_loss_even, diel_loss_odd, max_coupling, matched_impedance]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i]) for i in range(len(argout))]
    return arg

def edge_coupled_microstrip_analysis_view(arg, defaultunits):
    """
    """
    arg = arg[:11]
    newargs = convert2pq(arg, defaultunits)
    w, s, t, h, er, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    Z_even, Z_odd, eeff_even, eeff_odd = Z_eeff_edge_coupled_microstrip(w, er, t, h, s, freq)
    # visvis.clf()
    # diel = visvis.solidBox(translation=(0,0,h/t/2.0),scaling=(w/t*5.0,w/t*10.0,(h/t)))
    # line1 = visvis.solidBox(translation=(w/t/2+s/t/2,0,(h/t)+0.5),scaling=((w/t),w/t*10.0,(t/t)))
    # line2 = visvis.solidBox(translation=(-w/t/2-s/t/2,0,(h/t)+0.5),scaling=((w/t),w/t*10.0,(t/t)))
    # gnd = visvis.solidBox(translation=(0,0,-0.5),scaling=(w/t*5.0,w/t*10.0,(t/t)))
    # diel.faceColor="g"
    # line1.faceColor="r"
    # line2.faceColor="r"
    # gnd.faceColor="r"
    return

def edge_coupled_microstrip_synthesis(arg, defaultunits):
    r"""Synthesis function for the edge coupled microstrip transmission line.

    Args:
        arg(list): First 11 arguments are inputs.

            1. Line Width (w) ;length
            2. Line Gap (s) ;length
            3. Metal Thickness (t) ;length
            4. Substrate Thickness (h) ;length
            5. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            6. Dielectric Loss Tangent ;
            7. Metal Conductivity ;  electrical conductivity
            8. Metal Permeability ;
            9. Roughness ;length
            10. Frequency ; frequency
            11. Physical Length ;length
            12. Impedance (even);   impedance
            13. Impedance (odd);   impedance
            14. Electrical Length (even) ;   angle
            15. Electrical Length (odd) ;  angle
            16. <font size=+2>&epsilon;<sub>eff</sub></font> (even);
            17. <font size=+2>&epsilon;<sub>eff</sub></font> (odd);
            18. Conductor Loss (even) ;  loss per length
            19. Conductor Loss (odd) ;  loss per length
            20. Dielectric Loss (even) ; loss per length
            21. Dielectric Loss (odd) ; loss per length
            22. Maximum Coupling ;
            Ref: Transmssion Line Design Handbook, p199, with errata sheet
    """

    arg = arg[:14]
    newargs = convert2pq(arg, defaultunits)
    w, s, t, h, er, tand, sigma, mu, roughness, freq, length, Z_even, Z_odd, elec_length = tuple(newargs)
    output =Sentez(lambda *x:Z_eeff_edge_coupled_microstrip(*x)[:2], [w, er, t, h, s, freq], [0,4],target_value=[Z_even, Z_odd],init_value=[h,h], limits = [((h/100.0),h*10.0),((h/100.0),h*10.0)])
    w, s= tuple(output[0])
    Z_even, Z_odd, eeff_even, eeff_odd = Z_eeff_edge_coupled_microstrip(
        w, er, t, h, s, freq)
    # deg_even = electrical_length(eeff_even, freq, length)
    length = physical_length(eeff_even, freq, elec_length)
    deg_odd = electrical_length(eeff_odd, freq, length)
    sd = skin_depth(freq, sigma, mu)
    Z_even1, Z_odd1, eeff_even1, eeff_odd1 = Z_eeff_edge_coupled_microstrip(
        w - sd, 1.0001, t - sd, h + sd, s + sd, freq)
    Z_even2, Z_odd2, eeff_even2, eeff_odd2 = Z_eeff_edge_coupled_microstrip(
        w, 1.0001, t, h, s, freq)
    cond_loss_even = -pi * freq / Z_even / co * \
        (Z_even2 - Z_even1) * 20.0 * \
        log10(exp(1))  # dB/m, incremental inductance
    cond_loss_odd = -pi * freq / Z_odd / co * \
        (Z_odd2 - Z_odd1) * 20.0 * \
        log10(exp(1))  # dB/m, incremental inductance
    diel_loss_even = dielectric_loss(eeff_even, er, freq, tand)
    diel_loss_odd = dielectric_loss(eeff_odd, er, freq, tand)
    max_coupling = 20 * log10(((Z_even - Z_odd)/ (Z_even + Z_odd)))
    arg[0]=prettystring(w, defaultunits[0])
    arg[1]=prettystring(s, defaultunits[1])
    arg[10]=prettystring(length, defaultunits[10])
    argout = [ deg_odd, eeff_even, eeff_odd,
              cond_loss_even, cond_loss_odd, diel_loss_even, diel_loss_odd, max_coupling]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def Z_eeff_edge_coupled_microstrip(w, er, t, h, s, f):
    """
    Transmssion Line Design Handbook, p199, with errata sheet
    """
    g = (s/ h)
    u = (w/ h)
    fn = f * h * 1e-6
    v = u * (20.0 + g * g) / (10.0 + g * g) + g * exp(-g)
    a = 1.0 + (log(((v ** 4 + ((v/ 52.0)) ** 2)/ (v ** 4 + 0.432)))/ \
        49.0) + (log(1.0 + ((v/ 18.1)) ** 3)/ 18.7)
    b = 0.564 * (((er - 0.9)/ (er + 3.0))) ** 0.053
    eeff_0 = er_eff_qs_thick_microstrip(w, h, er, t)
    a0 = 0.7287 * (eeff_0 - 0.5 * (er + 1.0)) * (1.0 - exp(-0.179 * u))
    b0 = 0.747 * er / (0.15 + er)
    c0 = b0 - (b0 - 0.207) * exp(-0.414 * u)
    d0 = 0.593 + 0.694 * exp(-0.562 * u)
    eeff_0_odd = (0.5 * (er + 1.0) + a0 - eeff_0) * exp(-c0 * g ** d0) + eeff_0
    eeff_0_even = 0.5 * (er + 1.0) + 0.5 * (er - 1.0) * \
        (1.0 + (10.0/ v)) ** (-a * b)
    Q29 = (15.16/ (1.0 + 0.196 * (er - 1.0) ** 2))
    Q28 = 0.149 * (er - 1.0) ** 3 / (94.5 + 0.038 * (er - 1.0) ** 3)
    Q27 = 0.4 * g ** 0.84 * \
        (1.0 + 2.5 * (er - 1.0) ** 1.5 / (5.0 + (er - 1.0) ** 1.5))
    Q26 = 30.0 - (22.2/ (((13/ (er - 1.0))) ** 12 + 3.0)) - Q29
    Q25 = 0.3 * fn ** 2 / \
        (10.0 + fn ** 2) * (1.0 + (2.333/ ((5.0/ (er - 1.0) ** 2) + 1.0)))
    Q24 = 2.506 * Q28 * u ** 0.894 * \
        ((1.0 + 1.3 * u) * fn / 99.25) ** 4.29 / (3.575 + u ** 0.894)
    Q23 = 1.0 + 0.005 * fn * Q27 / \
        (1.0 + 0.812 * ((fn/ 15.0)) ** 1.9) / (1.0 + 0.025 * u * u)
    Q22 = 0.925 * ((fn/ Q26)) * 1.536 / (1.0 + 0.3 * ((fn/ 30)) ** 1.536)
    Q21 = abs(1.0 - 42.54 * g ** 0.133 * exp(-0.812)
              * g * u ** 2.5 / (1.0 + 0.033 * u ** 2.5))
    Q19 = 0.21 * g ** 4 / (1.0 + 0.18 * g ** 4.9) / \
        (1.0 + 0.1 * u * u) / (1.0 + ((fn/ 24.0)) ** 3)
    Q20 = Q19 * (0.09 + (1.0/ (1.0 + 0.1 * (er - 1.0) ** 2.7)))
    Q18 = 0.61 * (1.0 - exp(-2.13 * ((u/ 8)) ** 1.593)) / \
        (1.0 + 6.544 * g ** 4.17)
    Q17 = 0.394 * (1.0 - exp(-1.47 * ((u/ 7)) ** 0.672)) * \
        (1.0 - exp(-4.25 * ((fn/ 20)) ** 1.87))
    Q13 = 1.0 + 0.038 * ((er/ 8.)) ** 5.1
    Q14 = 1.0 + (1.203/ (((15/ er)) ** 4 + 1.0))
    Q15 = 1.887 * exp(-1.5 * g ** 0.84) * g ** Q14 / (1.0 + 0.41 *
                                                      ((fn/ 15)) ** 3 * u ** ((2./ Q13)) / (0.125 + u ** ((1.626/ Q13))))
    Q16 = Q15 * (1.0 + (9.0/ (1.0 + 0.403 * (er - 1.0) ** 2)))
    Q11 = 0.893 * (1.0 - (0.3/ (1.0 + 0.7 * (er - 1.0))))
    Q12 = 2.121 * ((fn/ 20.)) ** 4.91 * exp(-2.87 * g) * \
        g ** 0.902 / (1.0 + Q11 * ((fn/ 20.)) ** 4.91)
    r = ((fn/ 28.843)) ** 12.
    q = 0.016 + (0.0514 * er * Q21) ** 4.524
    p = 4.766 * exp(-3.228 * u ** 0.641)
    d = 5.086 * q * r / (0.3838 + 0.386 * q) * exp(-22.2 * u **
                                                   1.92) / (10 + 1.2992 * r) / ((1.0/ (er - 1.0) ** 6) + 10.0)
    c = 1.0 + 1.275 * (1.0 - exp(-0.004625 * p * er ** 1.674 *
                       ((fn/ 18.365)) ** 2.745)) - Q12 + Q16 - Q17 + Q18 + Q20
    Q1 = 0.8695 * u ** 0.194
    Q2 = 1.0 + 0.7519 * g + 0.189 * g ** 2.31
    Q3 = 0.1975 + (16.6 + ((8.4/ g)) ** 6.) ** (-0.387) + \
        (log((1./ ((1./ g ** 10) + (1./ 3.4 ** 10))))/ 241)
    Q4 = 2 * Q1 / Q2 / (exp(-g) * u ** Q3 + (2. - exp(-g)) * u ** (-Q3))
    Q5 = 1.794 + 1.14 * log(1. + (0.638/ (g + 0.517 * g ** 2.43)))
    Q6 = 0.2305 + (log((1./ ((1./ g ** 10) + (1./ 5.8 ** 10))))/ \
        281.3) + (log(1.0 + 0.598 * g ** 1.154)/ 5.1)
    Q7 = ((10.0 + 190.0 * g * g)/ (1.0 + 82.3 * g ** 3))
    Q8 = exp(-6.5 - 0.95 * log(g) - ((g/ 0.15)) ** 5)
    Q9 = log(Q7) * (Q8 + (1./ 16.5))
    Q10 = ((Q2 * Q4 - Q5 * exp(log(u) * Q6 * u ** (-Q9)))/ Q2)
    R1 = 0.03891 * er ** 1.4
    R2 = 0.267 * u ** 7.0
    R7 = 1.206 - 0.3144 * exp(-R1) * (1.0 - exp(-R2))
    R10 = 0.00044 * er ** 2.136 + 0.0184
    R11 = (1./ (((19.47/ fn)) ** 6. + 0.0962))
    R12 = (1.0/ (1.0 + 0.00245 * u * u))
    R15 = 0.707 * R10 * ((fn/ 12.3)) ** 1.097
    R16 = 1.0 + 0.0503 * er * er * R11 * (1.0 - exp(-((u/ 15.)) ** 6.))
    Q0 = R7 * (1.0 - 1.1241 * R12 / R16 * exp(-0.026 * fn ** 1.15656 - R15))
    P1 = 0.27488 + (0.6315 + (0.525/ (1.0 + 0.0157 * fn) ** 20)) * \
        u - 0.065683 * exp(-8.7513 * u)
    P2 = 0.33622 * (1.0 - exp(-0.03442 * er))
    P3 = 0.0363 * exp(-4.6 * u) * (1. - exp(-((fn/ 38.7)) ** 4.97))
    P4 = 1.0 + 2.751 * (1. - exp(-((er/ 15.916)) ** 8))
    P5 = 0.334 * exp(-3.3 * ((er/ 15)) ** 3) + 0.746
    P6 = P5 * exp(-((fn/ 18)) ** 0.368)
    P7 = 1.0 + 4.069 * P6 * g ** 0.479 * \
        exp(-1.347 * g ** 0.595 - 0.17 * g ** 2.5)
    P8 = 0.7168 * (1.0 + (1.076/ (1.0 + 0.0576 * (er - 1.0))))
    P9 = P8 - 0.7913 * (1.0 - exp(-((fn/ 20.)) ** 1.424)) * \
        arctan(2.481 * ((er/ 8.)) ** 0.946)
    P10 = 0.242 * (er - 1.0) ** 0.55
    P11 = 0.6366 * (exp(-3.401 * fn) - 1.0) * arctan(1.263 * ((u/ 3)) ** 1.629)
    P12 = P9 + ((1.0 - P9)/ (1.0 + 1.183 * u ** 1.376))
    P13 = 1.695 * P10 / (0.414 + 1.605 * P10)
    P14 = 0.8928 + 0.10722 * (1.0 - exp(-0.42 * ((fn/ 20.)) ** 3.215))
    P15 = abs(1.0 - 0.8928 * (1.0 + P11) * P12 * exp(-P13 * g ** 1.092) / P14)
    Fo = P1 * P2 * ((P3 * P4 + 0.1844) * fn * P15) ** 1.5763
    Fe = P1 * P2 * ((P3 * P4 + 0.1844 * P7) * fn) ** 1.5763
    eeff_odd = er - ((er - eeff_0_odd)/ (1. + Fo))
    eeff_even = er - ((er - eeff_0_even)/ (1. + Fe))
    Z = Z_qs_thick_microstrip(w, h, er, t)
    Ze = Z * csqrt((eeff_0/ eeff_0_even)) / \
        (1.0 - Z / eta0 * csqrt(eeff_0) * Q4)
    Zo = Z * csqrt((eeff_0/ eeff_0_odd)) / \
        (1.0 - Z / eta0 * csqrt(eeff_0) * Q10)
    Z_even = Ze * (0.9408 * (eeff_even) ** c - 0.9603) ** Q0 / \
        ((0.9408 - d) * (eeff_even) ** c - 0.9603) ** Q0
    Z = Z_disp_thick_microstrip(w, h, t, er, f)
    Z_odd = Z + ((Zo * ((eeff_odd/ eeff_0_odd)) ** Q22 - Z * Q23)/ \
        (1.0 + Q24 + (0.46 * g) ** 2.2 * Q25))
    return (Z_even, Z_odd, eeff_even, eeff_odd)

def Z_edge_coupled_thin_symmetric_stripline(w, b, s, er):
    """
    b:  ground spacing
    w:  line width
    s:  line spacing
    er: permittivity
    """
    # Transmssion Line Design Handbook, p232, g-yanduvarla hat arasi bosluk,
    # b-toplam yukseklik
    ke = tanh(pi * w / 2.0 / b) * tanh(pi * (w + s) / 2.0 / b)
    ko = (tanh(pi * w / 2.0 / b)/ tanh(pi * (w + s) / 2.0 / b))
    ke1 = csqrt(1.0 - ke * ke)
    ko1 = csqrt(1.0 - ko * ko)
    Zoe = 30.0 * pi / csqrt(er) * ellipk(ke1) / ellipk(ke)
    Zoo = 30.0 * pi / csqrt(er) * ellipk(ko1) / ellipk(ko)
    return (Zoe, Zoo)

def Z_edge_coupled_thick_symmetric_stripline(w, b, s, er, t):
    """
    b:  ground spacing
    w:  line width
    s:  line spacing
    er: permittivity
    t:  thickness
    Referans: Shielded Coupled-Strip Transmission Line.pdf
    """

    Z_even, Z_odd = Z_edge_coupled_thin_symmetric_stripline(w, b, s, er)
    Zst = Z_thick_stripline(w, b, t, er)
    Zs0 = Z_thick_stripline(w, b, 0.0, er)
    cf0 = 0.0885 * er * 2.0 * log(2.0) /  pi * (1e-4)
    if ( (t/ b) > 1.0e-5):
        cft = (1e-4) * 0.0885 * er /  pi * (2.0 / (1.0 - (t/ b)) * log((1.0/ (1.0 - (t/ b))) + 1.0) - \
                                          ((1.0/ (1.0 - (t/ b))) - 1.0) * log((1.0/ (1.0 - (t/ b)) ** 2) - 1.0))
    else:
        cft = (1e-4) * 0.0885 * er /  pi * (2.0 / (1.0 - (t/ b)) * log( 2.0 ))

    temp = (1.0/ Zst) - cft / cf0 * ((1.0/ Zs0) - (1.0/ Z_even))
    Zeven = (1.0/ temp)
    if (s/t) > 5.0:
        temp = (1.0/ Zst) + cft / cf0 * ((1.0/ Z_odd) - (1.0/ Zs0))
    else:
        temp = (1.0/ Z_odd) + ( (1.0/ Zst) - (1.0/ Zs0)) - 2.0 / eta0 * ( (cft/ er) - (cf0/ er) ) + 2.0 * t / eta0 / s
    Zodd = (1.0/ temp)
    return (Zeven, Zodd)

def edge_coupled_stripline_analysis(arg, defaultunits):
    """

    Args:
        arg(list): First 14 arguments are inputs.

            1. Line Width (w) ;length
            2. Line Spacing (s) ;length
            3. Metal Thickness ;length
            4. Ground Spacing (b) ;length
            5. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>) ;
            6. Dielectric Loss Tangent ;
            7. Metal Conductivity ;  electrical conductivity
            8. Metal Permeability ;
            9. Roughness ;length
            10. Frequency ; frequency
            11. Physical Length ;length
            12. Impedance (even);   impedance
            13. Impedance (odd);   impedance
            14. Electrical Length ;   angle
            15. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            16. Conductor Loss (Even Mode) ;  loss per length
            17. Conductor Loss (Odd Mode) ;  loss per length
            18. Dielectric Loss ; loss per length
            19. Maximum Coupling ;
            Ref: Transmssion Line Design Handbook, p233, with errata sheet
    """

    arg = arg[:11]
    newargs = convert2pq(arg, defaultunits)
    w, s, t, b, er, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    Z_even, Z_odd = Z_edge_coupled_thick_symmetric_stripline(w, b, s, er, t)
    sd = skin_depth(freq, sigma, mu)
    Z_even1, Z_odd1 = Z_edge_coupled_thick_symmetric_stripline(
        w - sd, b + sd, s + sd, 1.0, t)
    Z_even2, Z_odd2 = Z_edge_coupled_thick_symmetric_stripline(w, b, s, 1.0, t)
    eeff = er
    cond_loss_even = -pi * freq / Z_even / co * \
       (Z_even2 - Z_even1) * 20.0 * \
       log10(exp(1))  # dB/m, incremental inductance
    cond_loss_odd = -pi * freq / Z_odd / co * \
       (Z_odd2 - Z_odd1) * 20.0 * \
       log10(exp(1))  # dB/m, incremental inductance
    diel_loss = dielectric_loss(er, er, freq, tand)
    deg = electrical_length(eeff, freq, length)
    max_coupling = 20 * log10((fabs(Z_even - Z_odd)/ (Z_even + Z_odd)))
    argout = [Z_even, Z_odd, deg, eeff, cond_loss_even, cond_loss_odd , diel_loss, max_coupling]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def edge_coupled_stripline_analysis_view(arg, defaultunits):
    """
    """
    arg = arg[:11]
    newargs = convert2pq(arg, defaultunits)
    w, s, t, b, er, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    # visvis.clf()
    # diel = visvis.solidBox(translation=(0,0,b/t/2.0),scaling=(w/t*5.0,w/t*10.0,(b/t)))
    # line1 = visvis.solidBox(translation=(w/t/2+s/t/2,0,b/t/2.0),scaling=((w/t),w/t*10.1,(t/t)))
    # line2 = visvis.solidBox(translation=(-w/t/2-s/t/2,0,b/t/2.0),scaling=((w/t),w/t*10.1,(t/t)))
    # gnd1 = visvis.solidBox(translation=(0,0,-0.5),scaling=(w/t*5.0,w/t*10.0,(t/t)))
    # gnd2 = visvis.solidBox(translation=(0,0,(b/t)+0.5),scaling=(w/t*5.0,w/t*10.0,(t/t)))
    # diel.faceColor="g"
    # line1.faceColor="r"
    # line2.faceColor="r"
    # gnd1.faceColor="r"
    # gnd2.faceColor="r"
    return

def edge_coupled_stripline_synthesis(arg, defaultunits):
    """

    Args:
        arg(list): First 14 arguments are inputs.

            1. Line Width (w) ;length
            2. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>) ;
            3. Metal Thickness ;length
            4. Ground Spacing (b) ;length
            5. Line Spacing (s) ;length
            6. Dielectric Loss Tangent ;
            7. Metal Conductivity ;  electrical conductivity
            8. Metal Permeability ;
            9. Roughness ;length
            10. Frequency ; frequency
            11. Physical Length ;length
            12. Impedance (even);   impedance
            13. Impedance (odd);   impedance
            14. Electrical Length ;   angle
            15. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            16. Conductor Loss (Even Mode) ;  loss per length
            17. Conductor Loss (Odd Mode) ;  loss per length
            18. Dielectric Loss ; loss per length
            19. Maximum Coupling ;
            Ref: Transmssion Line Design Handbook, p233, with errata sheet
    """

    arg = arg[:14]
    newargs = convert2pq(arg, defaultunits)
    w, er, t, b, s, tand, sigma, mu, roughness, freq, length, Z_even, Z_odd, elec_length = tuple(newargs)
    output = Sentez(Z_edge_coupled_thick_symmetric_stripline, [w, b, s, er, t], [0,2], [Z_even, Z_odd] , [w,s] , [((b/100.0),10.0*b)])
    w, s= tuple(output[0])
    arg[0]=prettystring(w, defaultunits[0])
    arg[4]=prettystring(s, defaultunits[4])
    sd = skin_depth(freq, sigma, mu)
    Z_even1, Z_odd1 = Z_edge_coupled_thick_symmetric_stripline(w - sd, b + sd, s + sd, 1.0, t)
    Z_even2, Z_odd2 = Z_edge_coupled_thick_symmetric_stripline(w, b, s, 1.0, t)
    eeff = er
    cond_loss_even = -pi * freq / Z_even / co * (Z_even2 - Z_even1) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    cond_loss_odd = -pi * freq / Z_odd / co * (Z_odd2 - Z_odd1) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    diel_loss = dielectric_loss(er, er, freq, tand)
    max_coupling = 20 * log10((fabs(Z_even - Z_odd)/ (Z_even + Z_odd)))
    argout = [eeff, cond_loss_even, cond_loss_odd, diel_loss, max_coupling]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def Z_shielded_stripline(w, b, t, g, er):
    """Transmssion Line Design Handbook, p136, g-yanduvarla hat arasi bosluk,
    b-toplam yukseklik
    """
    cf0 = 1.0 * er * log(2.0) / pi
    cf = er / pi * (b / (b - t) * log(((2.0 * b - t)/ t)) +
                    log(t * (2.0 * b - t) / (b - t) ** 2.0))
    Zo = 30.0 * pi / csqrt(er) / (w / b / (1.0 - (t/ b)) + 2.0 *
                                  cf / pi / cf0 * log(1.0 + (1.0/ np.tan(pi * g / b))))
    return Zo

def conductor_loss_shielded_stripline(w, b, t, g, er, f, sigma, mu):
    """Incremental Inductance Rule"""
    sd = skin_depth(f, sigma, mu)
    z1 = Z_shielded_stripline(w - sd, b + sd, t - sd, g + 2 * sd, er)
    z2 = Z_shielded_stripline(w, b, t, g, er)
    return (mu * pi * f * sqrt(er) / co * np.abs(z1/ z2 - 1.0)) * 20.0 * log10(exp(1))

def symmetrical_shielded_stripline_analysis(arg, defaultunits):
    """Problemli.

    Args:
        arg(list): First 11 arguments are inputs.

            1. Line Width (w) ;length
            2. Ground Spacing (b);length
            3. Metal Thickness (t) ;length
            4. spacing between lateral wall and line (g) ;length
            5. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            6. Dielectric Loss Tangent ;
            7. Metal Conductivity ;  electrical conductivity
            8. Metal Permeability ;
            9. Roughness ;length
            10. Frequency ; frequency
            11. Physical Length ;length
            12. Impedance ;   impedance
            13. Electrical Length ;   angle
            14. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            15. Conductor Loss ;   loss per length
            16. Dielectric Loss ;  loss per length
            Ref: Transmssion Line Design Handbook, p136, g-yanduvarla hat arasi bosluk, b-toplam yukseklik, g<2b olmali
    """

    arg = arg[:11]
    newargs = convert2pq(arg, defaultunits)
    w, b, t, g, er, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    Z = Z_shielded_stripline(w, b, t, g, er)
    eeff = er
    deg = electrical_length(eeff, freq, length)
    cond_loss = conductor_loss_shielded_stripline( w, b, t, g, er, freq, sigma, mu)
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    argout = [Z, deg, eeff, cond_loss, diel_loss]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    if g > (2.0 * b):
        arg = arg + ["uyari g<2b olmali"]
    return arg

def symmetrical_shielded_stripline_synthesis(arg, defaultunits):
    """

    Args:
        arg(list): First 11 arguments are inputs.

            1. Line Width (w) ;length
            2. Ground Spacing (b);length
            3. Metal Thickness (t) ;length
            4. spacing between lateral wall and line (g) ;length
            5. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            6. Dielectric Loss Tangent ;
            7. Metal Conductivity ;  electrical conductivity
            8. Metal Permeability ;
            9. Roughness ;length
            10. Frequency ; frequency
            11. Physical Length ;length
            12. Impedance ;   impedance
            13. Electrical Length ;   angle
            14. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            15. Conductor Loss ;   loss per length
            16. Dielectric Loss ;  loss per length
            Ref: Transmssion Line Design Handbook, p136, g-yanduvarla hat arasi bosluk, b-toplam yukseklik, g<2b olmali
    """

    arg = arg[:13]
    newargs = convert2pq(arg, defaultunits)
    w, b, t, g, er, tand, sigma, mu, roughness, freq, length, Z, deg = tuple(newargs)
    output = Sentez(lambda *x: Z_shielded_stripline(*x), [w, b, t, g, er], [0], target_value=[Z],
                    init_value=[b], limits=[((b/ 1000.0), b * 1000.0)])
    w = output[0]
    Z = Z_shielded_stripline(w, b, t, g, er)
    eeff = er
    length = physical_length(eeff, freq, deg)
    cond_loss = conductor_loss_shielded_stripline( w, b, t, g, er, freq, sigma, mu)
    diel_loss = dielectric_loss(eeff, er, freq, tand)
    arg[0] = prettystring(w, defaultunits[0])
    arg[10] = prettystring(length, defaultunits[10])
    arg = arg[:11]
    argout = [Z, deg, eeff, cond_loss, diel_loss]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    if g > (2.0 * b):
        arg = arg + ["uyari g<2b olmali"]
    return arg

def broadside_offset_coupled_stripline_analysis(arg, defaultunits):
    """
    Ref: RF and Microwave Coupled Line Circuits

    Args:
        arg(list): First 11 arguments are inputs.

            1. Line Width (w) ;length
            2. Offset (wo) ;length
            3. Metal Thickness (t) ;length
            4. Spacing between lines (s) ;length
            5. Ground Spacing (b);length
            6. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            7. Dielectric Loss Tangent ;
            8. Metal Conductivity ;  electrical conductivity
            9. Metal Permeability ;
            10. Roughness ;length
            11. Frequency ; frequency
            12. Physical Length ;length
            13. Impedance (even) ;   impedance
            14. Impedance (odd);   impedance
            15. Electrical Length ;   angle
            16. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            17. Conductor Loss (Even Mode) ;  loss per length
            18. Conductor Loss (Odd Mode) ;  loss per length
            19. Dielectric Loss ; loss per length
            20. Maximum Coupling ;

    """

    arg = arg[:12]
    newargs = convert2pq(arg, defaultunits)
    w, wo, t, s, b, er, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    Z_even, Z_odd = Z_broadside_coupled_offset_stripline(w,wo,b,s,er)
    sd = skin_depth(freq, sigma, mu)
    Z_even1, Z_odd1 = Z_broadside_coupled_offset_stripline(
        w - sd, wo, b + sd, s + sd, 1.0)
    Z_even2, Z_odd2 = Z_broadside_coupled_offset_stripline(w, wo, b, s, 1.0)
    eeff = er
    cond_loss_even = -pi * freq / Z_even / co * (Z_even2 - Z_even1) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    cond_loss_odd = -pi * freq / Z_odd / co * (Z_odd2 - Z_odd1) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    diel_loss = dielectric_loss(er, er, freq, tand)
    deg = electrical_length(eeff, freq, length)
    max_coupling = 20.0 * log10((fabs(Z_even - Z_odd)/ (Z_even + Z_odd)))
    argout = [Z_even, Z_odd, deg, eeff, cond_loss_even, cond_loss_odd, diel_loss, max_coupling]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def broadside_offset_coupled_stripline_analysis_view(arg, defaultunits):
    """
    """
    arg = arg[:12]
    newargs = convert2pq(arg, defaultunits)
    w, wo, t, s, b, er, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    # visvis.clf()
    # diel = visvis.solidBox(translation=(0,0,0.0),scaling=((w+wo)/t*3.0,w/t*10.0,(b/t)))
    # line1 = visvis.solidBox(translation=(wo/t/2,0,s/t/2.0),scaling=((w/t),w/t*10.1,(t/t)))
    # line2 = visvis.solidBox(translation=(-wo/t/2,0,-s/t/2.0),scaling=((w/t),w/t*10.1,(t/t)))
    # gnd1 = visvis.solidBox(translation=(0,0,-b/t/2-0.5),scaling=((w+wo)/t*3.0,w/t*10.0,(t/t)))
    # gnd2 = visvis.solidBox(translation=(0,0,b/t/2+0.5),scaling=((w+wo)/t*3.0,w/t*10.0,(t/t)))
    # diel.faceColor="g"
    # line1.faceColor="r"
    # line2.faceColor="r"
    # gnd1.faceColor="r"
    # gnd2.faceColor="r"
    return

def broadside_offset_coupled_stripline_synthesis(arg, defaultunits):
    """

    Args:
        arg(list): First 11 arguments are inputs.

            1. Line Width (w) ;length
            2. Offset (wo) ;length
            3. Metal Thickness (t) ;length
            4. Spacing between lines (s) ;length
            5. Ground Spacing (b);length
            6. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            7. Dielectric Loss Tangent ;
            8. Metal Conductivity ;  electrical conductivity
            9. Metal Permeability ;
            10. Roughness ;length
            11. Frequency ; frequency
            12. Physical Length ;length
            13. Impedance (even) ;   impedance
            14. Impedance (odd);   impedance
            15. Electrical Length ;   angle
            16. <font size=+2>&epsilon;<sub>eff</sub></font> ;
            17. Conductor Loss (Even Mode) ;  loss per length
            18. Conductor Loss (Odd Mode) ;  loss per length
            19. Dielectric Loss ; loss per length
            20. Maximum Coupling ;
            Ref: RF and Microwave Coupled Line Circuits
    """

    arg = arg[:15]
    newargs = convert2pq(arg, defaultunits)
    w, b, t, s, wo, er, tand, sigma, mu, roughness, freq, length, Z_even, Z_odd, elec_length = tuple(newargs)
    w , wo = width_broadside_coupled_offset_stripline(Z_even,Z_odd,b,s,er)
    sd = skin_depth(freq, sigma, mu)
    Z_even1, Z_odd1 = Z_broadside_coupled_offset_stripline(
        w - sd, wo, b + sd, s , 1.0)
    Z_even2, Z_odd2 = Z_broadside_coupled_offset_stripline(w, wo, b, s, 1.0)
    eeff = er
    cond_loss_even = -pi * freq / Z_even / co * (Z_even2 - Z_even1) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    cond_loss_odd = -pi * freq / Z_odd / co * (Z_odd2 - Z_odd1) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    diel_loss = dielectric_loss(er, er, freq, tand)
    length = physical_length(eeff, freq, elec_length)
    arg[11]= prettystring(length, defaultunits[11])
    arg[0] = prettystring(w, defaultunits[11])
    arg[4] = prettystring(wo, defaultunits[11])
    max_coupling = 20.0 * log10((fabs(Z_even - Z_odd)/ (Z_even + Z_odd)))
    argout = [eeff, cond_loss_even, cond_loss_odd, diel_loss, max_coupling]
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def width_broadside_coupled_offset_stripline(Zeven,Zodd,b,s,er):
    s=(s/b)
    Zo=csqrt(Zeven*Zodd)
    cc=(Zeven/Zodd)
    Co=120.0*pi*csqrt((cc/er))/Zo
    A=exp(60.0*pi**2/csqrt(er)/Zo*(((1.0-cc*s)/csqrt(cc))))
    A=np.max((A,4.0))
    B=0.5*(A-2.0+sqrt(fabs(A*A-4.0*A)))
    p=0.25*(B-1.0)*(1.0+s)+0.25*sqrt((B-1.0)**2*(1.0+s)**2+16.0*s*B)
    r=s*B/p
    Cf0=1.0/pi*(-2.0/(1.0-s)*log(s)+0.5/s*log(p*r/((p+s)*(1.0+p)*(r-s)*(1.0-r))))
    w=0.5*s*(1.0-s)*(Co-Cf0)
    wo=0.5/pi*((1.0+s)*log((p/r))+(1.0-s)*log((1.0+p)*(r-s)/(s+p)/(1.0-r)))
    if ((w/(1.0-s))<0.35): # loose coupling
        dC=120.0*pi*(cc-1.0)/csqrt(er)/Zo/csqrt(cc)
        K=(1.0/(exp(pi*dC/2.0)-1.0))
        a=csqrt((((s-K)/(s+1.0)))**2+K)-((s-K)/(s+1.0))
        q=(K/a)
        Cf0=0.5/pi*(0.5/(1.0+s)*log((1.0+a)/a/(1.0-q))-1.0/(1.0-s)*log(q))
        wc=1.0/pi*(s*log((q/a))+(1.0-s)*log(((1.0-q)/(1.0+a))))
        Cf=-2.0/pi*(1.0/(1.0+s)*log(0.5*(1.0-s))+1.0/(1.0-s)*log(0.5*(1.0+s)))
        w=0.25*(1.0-s*s)*(Co-Cf0-Cf)
        wo=w-wc
    w=w*b
    wo=wo*b
    return (w,wo)

def Z_broadside_coupled_offset_stripline(w,wo,b,s,er):
    output =Sentez(width_broadside_coupled_offset_stripline, [(b/2.0), (b/10.0), b,s,er], [0,1],target_value=[w,wo],init_value=[60,40], limits = [(5.0,200.0),(5.0,200.0)])
    return output[0] #(Zeven,Zodd)

def broadside_coupled_suspended_stripline_analysis(arg, defaultunits):
    """

    Args:
        arg(list): First 11 arguments are inputs.

            1. Line Width (w) ;length
            2. Ground Spacing (b);length
            3. Metal Thickness (t) ;length
            4. Spacing between lines (s) ;length
            5. Dielectric Permittivity (<font size=+2>&epsilon;<sub>r</sub></font>);
            6. Dielectric Loss Tangent ;
            7. Metal Conductivity ;  electrical conductivity
            8. Metal Permeability ;
            9. Roughness ;length
            10. Frequency ; frequency
            11. Physical Length ;length
            12. Impedance (even) ;   impedance
            13. Impedance (odd);   impedance
            14. Electrical Length (Even Mode);   angle
            15. Electrical Length (Odd Mode) ;   angle
            16. <font size=+2>&epsilon;<sub>eff</sub> (even)</font> ;
            17. <font size=+2>&epsilon;<sub>eff</sub> (odd)</font> ;
            18. Conductor Loss (Even Mode) ;  loss per length
            19. Conductor Loss (Odd Mode) ;  loss per length
            20. Dielectric Loss (Even Mode); loss per length
            21. Dielectric Loss (Odd Mode) ; loss per length
            22. Maximum Coupling ;

            Ref: RF and Microwave Coupled Line Circuits
    """

    arg = arg[:11]
    newargs = convert2pq(arg, defaultunits)
    w, b, t, s, er, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    Z_even,Z_odd,eeff_even,eeff_odd = Z_eeff_broadside_coupled_suspended_stripline(w,s,b,er)
    sd = skin_depth(freq, sigma, mu)
    Z_even1, Z_odd1, eeff_even1, eeff_odd1 = Z_eeff_broadside_coupled_suspended_stripline(
        w - sd, s, b + sd, 1.0 )
    Z_even2, Z_odd2, eeff_even2, eeff_odd2 = Z_eeff_broadside_coupled_suspended_stripline(w, s, b, 1.0)

    cond_loss_even = -pi * freq / Z_even / co * (Z_even2 - Z_even1) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    cond_loss_odd = -pi * freq / Z_odd / co * (Z_odd2 - Z_odd1) * 20.0 * log10(exp(1))  # dB/m, incremental inductance
    diel_loss_even = dielectric_loss(eeff_even, er, freq, tand)
    diel_loss_odd = dielectric_loss(eeff_odd, er, freq, tand)
    deg_even = electrical_length(eeff_even, freq, length)
    deg_odd = electrical_length(eeff_odd, freq, length)
    max_coupling = 20.0 * log10((fabs(Z_even - Z_odd)/ (Z_even + Z_odd)))
    argout = [Z_even, Z_odd, deg_even, deg_odd, eeff_even, eeff_odd, cond_loss_even, cond_loss_odd, diel_loss_even, diel_loss_odd,max_coupling]
    print("df ",len(arg),len(argout),len(defaultunits))
    arg = arg + [prettystring(argout[i], defaultunits[len(arg) + i])
                 for i in range(len(argout))]
    return arg

def broadside_coupled_suspended_stripline_analysis_view(arg, defaultunits):
    """
    """
    arg = arg[:11]
    newargs = convert2pq(arg, defaultunits)
    w, b, t, s, er, tand, sigma, mu, roughness, freq, length = tuple(newargs)
    # visvis.clf()
    # diel = visvis.solidBox(translation=(0,0,0.0),scaling=((w)/t*5.0,w/t*10.0,(s/t)))
    # line1 = visvis.solidBox(translation=(0,0,s/t/2.0+t/t/2.0),scaling=((w/t),w/t*10.1,(t/t)))
    # line2 = visvis.solidBox(translation=(0,0,-s/t/2.0-t/t/2.0),scaling=((w/t),w/t*10.1,(t/t)))
    # gnd1 = visvis.solidBox(translation=(0,0,-b/t/2-0.5),scaling=((w)/t*5.0,w/t*10.0,(t/t)))
    # gnd2 = visvis.solidBox(translation=(0,0,b/t/2+0.5),scaling=((w)/t*5.0,w/t*10.0,(t/t)))
    # diel.faceColor="g"
    # line1.faceColor="r"
    # line2.faceColor="r"
    # gnd1.faceColor="r"
    # gnd2.faceColor="r"
    return

def microstrip_step_in_width(w1, w2, eps_r, h, t, freq):
    """ Reference: Transmission Line Design Handbook p. 317"""
    if abs(w2-w1)>w1*0.01:
        eeff1 = er_eff_disp_thick_microstrip(w1, h, t, eps_r, freq)
        eeff2 = er_eff_disp_thick_microstrip(w2, h, t, eps_r, freq)
        Z1 = Z_disp_thick_microstrip(w1, h, t, eps_r, freq)
        Z2 = Z_disp_thick_microstrip(w2, h, t, eps_r, freq)
        C = sqrt(w1*w2)*((10.1*log10(eps_r)+2.33)*w1/w2-12.6*log10(eps_r)-3.17)*1e-12
        Ls = h*(40.5*(w1/w2-1)-75*log10(w1/w2)+0.2*(w1/w2-1)**2)*1e-9
        Lw1 = Z1*sqrt(eeff1)/co
        Lw2 = Z2*sqrt(eeff2)/co
        L1 = Lw1*Ls/(Lw1+Lw2)
        L2 = Lw2*Ls/(Lw1+Lw2)
        ABCDmatrix = [  -4*pi*pi*freq*freq*L1*C+1,  2j*pi*freq*(L1+L2)-1j*(2*pi*freq)**3*L1*L2*C,
                        2j*pi*freq*C,               -(2*pi*freq)**2*L2*C+1]
    else:
        ABCDmatrix = [  1,  0,
                        0,  1]
    return ABCDmatrix # [A, B, C, D] list

def stripline_step_in_width2(w1, w2, eps_r, h1, h2, t, freq):
    """ Reference: Transmission Line Design Handbook p. 350
    DOES NOT WORK, may be problems in units"""
    w1, w2 = min(w1,w2), max(w1,w2)
    if abs(w2-w1)>w1*0.01:
        Z1 = Z_thick_offset_stripline(w1, eps_r, h1, h2, t)
        Z2 = Z_thick_offset_stripline(w2, eps_r, h1, h2, t)
        b = h1 + h2 + t
        d_= w1+2*b/pi*log(2.0)
        d = w2+2*b/pi*log(2.0)
        alpha = d_/d
        k = tanh(pi*w1/2/b)
        la = 2*pi/k
        lg = co/freq/sqrt(eps_r)
        A = ((1+alpha)/(1-alpha))**(2*alpha)*((1+sqrt(1-d**2/la**2))/(1-sqrt(1-d**2/la**2)))-(1+3*alpha**2)/(1-alpha**2)
        KK = log(((1-alpha**2)/(4*alpha))*((1+alpha)/(1-alpha))**((1+alpha)/(2*alpha)))+2/A
        L = Z1*2*d*KK/lg
        ABCDmatrix  = [ 1, 1j*(2*pi*freq)*L,
                        0, 1]
    else:
        ABCDmatrix  = [ 1, 0,
                        0, 1]
    return ABCDmatrix # [A, B, C, D] list

def stripline_step_in_width(w1, w2, eps_r, h1, h2, t, freq):
    """ Reference: Transmission Line Design Handbook p. 350"""
    w1, w2 = min(w1,w2), max(w1,w2)
    if abs(w2-w1)>w1*0.01:
        Z1 = Z_thick_offset_stripline(w1, eps_r, h1, h2, t)
        Z2 = Z_thick_offset_stripline(w2, eps_r, h1, h2, t)
        b = h1 + h2 + t
        d1 = w1+2*b/pi*log(2.0)
        lg = co/freq/sqrt(eps_r)
        L = Z1*2*d1/lg*log(1/sin(pi*Z2/2/Z1))/(2*pi*mean(freq))

        length= b*log(2)/pi
        ct = cos(2*pi/lg*length/mean(freq)*freq)
        st = sin(2*pi/lg*length/mean(freq)*freq)
        w=2*pi*freq
        X=1j*w*L
        ABCDmatrix  = [ st**2*Z2/Z1+ct*(ct-1j*X*st/Z1), 1j*ct*st*Z2+ct*(X*ct-1j*st*Z1),
                        1j*st*(ct-1j*X*st/Z1)/Z2-1j*ct*st/Z1, 1j*st*(X*ct-1j*st*Z1)/Z2+ct**2]
    else:
        ABCDmatrix  = [ 1, 0,
                        0, 1]
    return ABCDmatrix # [A, B, C, D] list

def Z_eeff_broadside_coupled_suspended_stripline(w,s,b,er):
    """
    Ref: RF and Microwave Coupled-Line Circuits
    """
    Zo1=60.0*log(3.0*s/w+sqrt(((s/w))**2+1))
    P=270.0*(1.0-tanh(0.28+1.2*sqrt((b/s)-1.0)))
    Q=1.0-arctanh(((0.48*sqrt(2.0*w/s-1.0))/((b/s))**2))
    if (w/s)<0.5:
        dZ=P
    else:
        dZ=P*Q
    Zodd=Zo1-dZ
    k=tanh(293.9*s/b/Zodd/sqrt(er))
    Zeven=eta0/2.0*ellipk(csqrt(1.0-k*k))/ellipk(k)

    ber=0.564*power(((er-0.9)/(er+3.0)),0.053)
    u=2.0*w/s
    a=1.0+1.0/49.0*log(((u**4+((u/52.0))**2)/(u**4+0.432)))+1.0/18.7*log(1.0+((u/18.1))**3)
    qinf=power(1.0+5.0*s/w,a*ber)
    qc=tanh(1.043+0.121*((b/s)-1.0)-1.164*s/(b-s))
    q=qinf*qc
    eeff_odd=0.5*(er+1.0)+0.5*q*(er-1.0)

    a1=(0.8145-0.05824*log((s/b)))**8
    b1=(0.7581-0.07143*log((s/b)))**8
    eeff_even=(1.0+s/b*(a1-b1*log((w/b)))*(sqrt(er)-1.0))**2

    return (Zeven,Zodd,eeff_even,eeff_odd)

# microstrip_analysis = microstrip_analysis
# stripline_analysis = stripline_analysis
edge_coupled_stripline_analysis = edge_coupled_stripline_analysis
broadside_coupled_stripline_analysis = broadside_offset_coupled_stripline_analysis
#symmetrical_shielded_stripline_analysis = shielded_stripline_analysis
suspended_icrostripline_analysis = suspended_microstripline_analysis
shielded_suspended_stripline_analysis = shielded_suspended_stripline_analysis
grounded_coplanar_waveguide_analysis = grounded_cpw_analysis
#covered_grounded_coplanar_waveguide_analysis = covered_grounded_cpw_analysis
eccentric_coaxial_line_analysis = eccentric_coaxial_analysis
#rectangular_coaxial_line_analysis = rectangular_coaxial_analysis
coaxial_line_strip_center_analysis = coaxial_strip_center_analysis
square_coaxial_line_circular_center_analysis = square_coaxial_circular_center_analysis
#square_coaxial_line_square_center_analysis = square_coaxial_square_center_analysis
parallel_wires_analysis = parallel_wires_analysis
edge_coupled_microstrip_analysis = edge_coupled_microstrip_analysis

# microstrip_synthesis = microstrip_synthesis
# stripline_synthesis = stripline_synthesis
# shielded_suspended_stripline_synthesis = shielded_suspended_stripline_synthesis
#symmetrical_shielded_stripline_synthesis = shielded_stripline_synthesis
suspended_microstripline_synthesis = suspended_microstripline_synthesis
grounded_coplanar_waveguide_synthesis = grounded_cpw_synthesis
edge_coupled_microstrip_synthesis = edge_coupled_microstrip_synthesis
edge_coupled_stripline_synthesis = edge_coupled_stripline_synthesis
broadside_coupled_stripline_synthesis = broadside_offset_coupled_stripline_synthesis

if __name__ == "__main__":
    # arg =["55mil","0.1mil","5mil", "20mil", "20mil", "2.2","0.001","10e9","1.0","0.0","10GHz","100mil","50","100"]
    # args=covered_suspended_microstripline_analysis(arg,[""]*17)
    # print(args)
    # print(Z_eeff_suspended_stripline(55, 0.1, 5, 20, 20, 2.2, 1000))
    # print(Z_eeff_grounded_cpw_thick(100e-6, 0.1e-6, 3.0, 100e-6, 127e-6))
    # print(Z_coaxial(3.4, 80 , 140))
    print(stripline_step_in_width(130e-6, 200e-6, 3, 100e-6, 300e-6, 30e-6, 77e9))
    print(stripline_step_in_width2(130e-6, 200e-6, 3, 100e-6, 300e-6, 30e-6, 77e9))
