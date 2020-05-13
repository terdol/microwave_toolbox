#-*-coding:utf-8-*-
"""
General methods
"""
import sys
import quantities as pq
import numpy as np
import re
import collections
import cProfile
from copy import deepcopy


Np2dB = 8.68589 # guc kaybi icin alpha'dan' dB/m'ye donusum katsayisi (20.0*log10(e))
                # alpha'nin birimi 1/m (Neper)'dir
globsep = ";"
globsep2 = ":"  # empedans degerlerini ayirmak icin
### New Units For Quantities Library
pq.mH = pq.millihenry = pq.UnitQuantity('millihenry', pq.H / 1000, symbol='mH')
pq.uH = pq.microhenry = pq.UnitQuantity('microhenry', pq.H / 1e6, symbol='uH')
pq.nH = pq.nanohenry = pq.UnitQuantity('nanohenry', pq.H / 1e9, symbol='nH')
pq.pH = pq.picohenry = pq.UnitQuantity('picohenry', pq.H / 1e12, symbol='pH')

pq.mF = pq.millifarad = pq.UnitQuantity('millifarad', pq.F / 1000, symbol='mF')
pq.uF = pq.microfarad = pq.UnitQuantity('microfarad', pq.F / 1e6, symbol='uF')
pq.nF = pq.nanofarad = pq.UnitQuantity('nanofarad', pq.F / 1e9, symbol='nF')
pq.pF = pq.picofarad = pq.UnitQuantity('picofarad', pq.F / 1e12, symbol='pF')

pq.microinch = pq.UnitQuantity('microinch', pq.inch / 1e6, symbol='microinch')

pq.dB = pq.UnitQuantity('dB', pq.dimensionless, symbol='dB')

def ekpolyfit(x):
    #polynomial fit for ellipk function. works from 0 to 0.98 with good accuracy.
    p1 =       36.94 
    p2 =        -114 
    p3 =       148.9 
    p4 =        -106 
    p5 =       44.75 
    p6 =      -11.16 
    p7 =       1.807 
    p8 =     0.09132 
    p9 =      0.3972 
    p10 =       1.571
    ek = p1*x**9+p2*x**8+ p3*x**7 + p4*x**6 +p5*x**5 + p6*x**4 + p7*x**3 + p8*x**2 + p9*x + p10
    return ek

class dotdict(dict):
    """dot.notation access to dictionary attributes
    This class is not pickleable!!! 
    objdict is pickleable, because it raises correct exceptions,
    dill instead of pickle does not work too.
    """
    # __getattr__ = dict.get
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

class objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def tukey_window(alpha,N):
    """
    Tukey window (also known as "tapered cosine window")
    Also available in scipy.signal
    """
    sonuc=[]
    cos = np.cos
    pi = np.pi
    for i in range(N):
        if (i<=alpha*(N-1.0)/2.0):
            sonuc.append(0.5*(1.0+cos(pi*(2.0*i/alpha/(N-1.0)-1.0))))
        elif (i<=(N-1)*(1.0-alpha/2.0)):
            sonuc.append(1.0)
        elif (i<=(N-1)):
            sonuc.append(0.5*(1.0+cos(pi*(2.0*i/alpha/(N-1.0)-2.0/alpha+1.0))))
    return sonuc

def blackman_window(N):
    """
    Blackman-Harris window
    Also available in scipy.signal
    """
    sonuc=[]
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168
    for i in range(N):
        sonuc.append(a0-a1*np.cos(2*np.pi*i/(N-1))+a2*np.cos(4*np.pi*i/(N-1))-a3*np.cos(6*np.pi*i/(N-1)))
        # sonuc.append(0.42+0.5*np.cos(np.pi*i/(N-1))+0.08*np.cos(2*np.pi*i/(N-1)))
    return np.array(sonuc)

def gaussian_window(sigma,N):
    """
    Gaussian window
    sigma should be smaller than or equal to 0.5
    Also available in scipy.signal
    Ref: Wikipedia
    """
    sonuc=[]
    exp = np.exp
    for i in range(N):
        sonuc.append(exp(-0.5*((i-(N-1.0)/2.0)/(sigma*(N-1.0)/2.0))**2))
    return sonuc
    
def cmp(x, y):
    """
    Replacement for built-in function cmp that was removed in Python 3

    Compare the two objects x and y and return an integer according to
    the outcome. The return value is negative if x < y, zero if x == y
    and strictly positive if x > y.
    """
    return int(x > y) - int(x < y)

try:
    basestring
except NameError:
    basestring = str
    
try:
    unicode
except NameError:
    unicode = bytes

def do_cprofile(func):
    # """ Bu fonksiyon istenen fonksiyona profiling yapmayi saglayan decoratordur
        # ornek kullanim:
        # @do_cprofile
        # def expensive_function():
            # for x in get_number():
                # i = x ^ x ^ x
            # return 'some result!'

        #sperform profiling
        # result = expensive_function()  
        
        # Referans: https://zapier.com/engineering/profiling-python-boss/
    # """
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func  

# Line Profiler ile profiling

# Kullanim:
# @do_profile(follow=[get_number])
# def expensive_function():
    # for x in get_number():
        # i = x ^ x ^ x
    # return 'some result!'
# Handy tip: Just decorate your test function and pass the problem function in the follow argument!
# Referans: https://zapier.com/engineering/profiling-python-boss/

try:
    from line_profiler import LineProfiler

    def do_profile(follow=[]):
        def inner(func):
            def profiled_func(*args, **kwargs):
                try:
                    profiler = LineProfiler()
                    profiler.add_function(func)
                    for f in follow:
                        profiler.add_function(f)
                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    profiler.print_stats()
            return profiled_func
        return inner

except ImportError:
    def do_profile(follow=[]):
        "Helpful if you accidentally leave in production!"
        def inner(func):
            def nothing(*args, **kwargs):
                return func(*args, **kwargs)
            return nothing
        return inner

def printall(isimler, args):
    k = isimler.split(",")
    for i in range(len(args)):
        print(k[i] + " = " + str(args[i]))

def prettystring(miktarlar, birim=None):
    formatstring = '{:3.7e}'
    if len(np.shape(miktarlar)) == 0:
        miktarlar = [miktarlar]
    liste = []
    for i in range(len(miktarlar)):
        miktar = miktarlar[i]
        if (np.fabs(float(miktar))==np.inf):
            liste.append(str(miktar))
            continue
        if isinstance(miktar, pq.Quantity):
            if birim == "SI":
                miktar = miktar.simplified
                dims = miktar.dimensionality.string
                miktar = miktar.magnitude
            elif birim == "dBm":
                temp = miktar.simplified.magnitude
                miktar = 10.0 * np.log10(temp * 1000.0)
                dims = birim
            elif birim is not None:
                miktar.units = birim
                miktar = miktar.magnitude
                dims = birim
            else:
                miktar = miktar.magnitude
                dims = miktar.dimensionality.string
            if dims == "dimensionless":
                dims = ""
            liste.append((formatstring.format(float(miktar)) + " " + dims).strip())
        elif isinstance(miktar, (float, int, np.ndarray)):
            miktar = float(miktar)
            if birim == "dBm":
                miktar = 10.0 * np.log10(miktar * 1000.0)
                dims = birim
            elif birim is not None:
                miktar = pq.Quantity(miktar * coef(birim), birim).magnitude
                dims = birim
            else:
                dims = ""
            if dims == "dimensionless":
                dims = ""
            miktar = float(miktar)
            # print "miktar= ",miktar,type(miktar)
            liste.append((formatstring.format(miktar) + " " + dims).strip())
        elif isinstance(miktar, str):
            temp = convert2pq(miktar)[0]
            if birim is not None:
                temp = pq.Quantity(temp * coef(birim), birim)
                dims = birim
                liste.append((formatstring.format(float(temp.magnitude)) + " " + dims).strip())
            else:
                liste.append(miktar)
    return globsep.join(liste)

def coef(birim):
    """ SI'dan Birim'e cevirmek icin katsayi """
    temp = pq.Quantity(1.0, birim)
    return (temp.magnitude) / (temp.simplified.magnitude)

def convert2pq(sayilar, defaultunits=[]):
    """
    Method to convert a string or string list to float after unit conversion to SI
    Units are extracted from strings. 
    If there is not a unit in string, unit is taken from defaultunits
    """
    
    number = ""
    # unit = ""
    if len(defaultunits)==0:
        defaultunits=[""]*len(sayilar)    
    sonuc = []  
       
    if not hasattr(convert2pq, "sayilar"):   # make sayilar static variable  
        convert2pq.sayilar = deepcopy(sayilar)
        convert2pq.defaultunits=deepcopy(defaultunits)
    elif (convert2pq.sayilar == sayilar) and convert2pq.defaultunits == defaultunits:
        return convert2pq.sonuc   

    if isinstance(sayilar, (float,int)):
        return [float(sayilar)]
    if isinstance(sayilar, str):
        sayilar = [sayilar]  
               
#    if not hasattr(convert2pq, "regex"):   
         # make regex static variable, did not provide any performance advantage
#        # pattern for matching real numbers
#        pattern = r"([+\-]?)(\d+(\.\d*)?|\d*\.\d+)([eE][+\-]?\d+)?\s*(\D+\S*)?"
#        convert2pq.regex = re.compile(pattern) 
        
    sonuclar = []    
        
    for i in range(len(sayilar)):
        sayi = sayilar[i] 
        try:
          if (sayi == convert2pq.sayilar[i]) and (convert2pq.units[i]):
              sonuclar.append(convert2pq.sonuc[i])
              continue
        except:
            pass
        #print "sss ",i, " ",sayi
        if isinstance(sayi, pq.Quantity):
            sonuc = sayi
            sonuclar.append(float(sonuc.simplified.magnitude)) 
        elif isinstance(sayi, (float, np.ndarray, int, list)):
            # unit = ""
            # if len(defaultunits) > 0:
                # unit = defaultunits[i]
            # sonuc=pq.Quantity(float(sayi), unit) 
            print(sayi)
            print(np.array(sayi))
            print(np.shape(np.array(sayi)))
            sonuc=pq.Quantity(np.array(sayi), defaultunits[i]) 
            sonuclar.append(sonuc.simplified.magnitude) 
        else:
            sayi = sayi.split("=")[-1].strip()        
            #match = convert2pq.regex.search(sayi) 
            match = re.search(r"([+\-]?)(\d+(\.\d*)?|\d*\.\d+)([eE][+\-]?\d+)?\s*(\D+\S*)?",sayi)
            number = ""
            unit = ""
            if match is not None:
                for k in [1, 2, 4]:
                    if match.group(k) is not None:
                        number = number + str(match.group(k))
                unit = str(match.group(5))
                if unit == "None":
                    # unit = ""
                    # if len(defaultunits) > 0:
                        # unit = defaultunits[i]
                    unit = defaultunits[i]
                sonuc = pq.Quantity(float(number), unit)
            else:
                sonuc = None
            sonuclar.append(float(sonuc.simplified.magnitude)) 
    convert2pq.sonuc = deepcopy(sonuclar)
    convert2pq.sayilar = deepcopy(sayilar)
    convert2pq.defaultunits=deepcopy(defaultunits)
    return convert2pq.sonuc

def flatten(x):
    """Flatten (an irregular) list of lists"""
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def flatten2(l):
    """Flatten (an irregular) list of lists (yield version of flatten)"""
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten2(el):
                yield sub
        else:
            yield el

def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal        
    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)    
    see also:     
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """

    import numpy as np

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':  #moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]


    
from numpy import NaN, Inf, arange, isscalar, asarray, array

def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)


if __name__ == "__main__":
    print(convert2pq("10mil"))
    print(convert2pq(["10", "10 inch"], ["m", "m"]))
    print(convert2pq([10.0, "10 inch"]))
    print(convert2pq([[10.0, 36.2], "10 inch"], ["mil", "m"]))
    a = np.array([12.1, 45.3])
    print(prettystring(a, birim=""))

   # arg = ["12.1", "34.2", "23.4"]
   # arg = [np.array([12.1, 15.2, 18.3]), "34.2", "23.4"]
   # arg1 = Tee_Attenuator_Analysis(arg, ["ohm", "ohm", "ohm", "", "", "", "", ""])  
    # import time
    # time.clock()
    # for i in np.linspace(10,50,10000):
        # # convert2pq([i,"20","1","0","0","0","0","0","0","0","0"],[])            
        # convert2pq([i,"20"],[])            
    # print time.clock()       
    # for i in range(10,50,1000):
        # convert2pq_eski([10,"20","1","0","0","0","0","0","0","0","0"],[]) 

    # print convert2pq([10.0], defaultunits=["GHz"])    
    # print convert2pq([10.0], defaultunits=["GHz"])    
    # print convert2pq([10.0], defaultunits=["Hz"])    
    # print convert2pq([10.0], defaultunits=["MHz"])    
    # print convert2pq([100.0], defaultunits=["MHz"])   
    pass

    