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
import os
from functools import lru_cache

Np2dB = 8.68589 # coefficient to convert from alpha to dB/m (20.0*log10(e))
                # unit of alpha is 1/m (Neper)
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
    """Polynomial fit for ellipk function. works from 0 to 0.98 with good accuracy."""
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
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168
    arr = np.linspace(0,N-1,N)
    return a0-a1*np.cos(2*np.pi*arr/(N-1))+a2*np.cos(4*np.pi*arr/(N-1))-a3*np.cos(6*np.pi*arr/(N-1))

def gaussian_window(sigma,N):
    """
    Gaussian window
    sigma should be smaller than or equal to 0.5
    Also available in scipy.signal
    Ref: Wikipedia
    """
    arr = np.linspace(0,N-1,N)
    return np.exp(-0.5*((arr-(N-1.0)/2.0)/(sigma*(N-1.0)/2.0))**2)

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


def prettystring(miktarlar, birim=None):
    formatstring1 = '{:3.3e}'
    formatstring2 = '{:3.3f}'
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
            miktar = float(miktar)
            if miktar<0.001:
                formatstring = formatstring1
            else:
                formatstring = formatstring2
            liste.append((formatstring.format(miktar) + " " + dims).strip())
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
            if miktar<0.001:
                formatstring = formatstring1
            else:
                formatstring = formatstring2
            liste.append((formatstring.format(miktar) + " " + dims).strip())
        elif isinstance(miktar, str):
            temp = convert2pq(miktar)[0]
            if birim is not None:
                temp = pq.Quantity(temp * coef(birim), birim)
                dims = birim
                if float(temp.magnitude)<0.001:
                    formatstring = formatstring1
                else:
                    formatstring = formatstring2
                liste.append((formatstring.format(float(temp.magnitude)) + " " + dims).strip())
            else:
                liste.append(miktar)
    return globsep.join(liste)

def coef(outputunit):
    """ Coefficient to convert from SI unit to outputunit """
    temp = pq.Quantity(1.0, outputunit)
    return (temp.magnitude) / (temp.simplified.magnitude)

# remove from this module
def stripunit(sayi):
    match = re.search(r"([+\-]?)(\d+(\.\d*)?|\d*\.\d+)([eE][+\-]?\d+)?\s*(\D+\S*)?",sayi)
    number = ""
    unit = ""
    if match is not None:
        for k in [1, 2, 4]:
            if match.group(k) is not None:
                number = number + str(match.group(k))
    return float(number)

# @lru_cache did not work with arguments of type list since lists are not hashable
def convert2pq(sayilar, defaultunits=None):
    """
    Method to convert a string or string list to float after unit conversion to SI
    Units are extracted from strings.
    If there is not a unit in string, unit is taken from defaultunits
    """

    number = ""
    # unit = ""
    if isinstance(sayilar, (float,int)):
        return [float(sayilar)]
    if isinstance(sayilar, str):
        sayilar = [sayilar]
    if not defaultunits:
        defaultunits=[""]*len(sayilar)
    if isinstance(defaultunits, str):
        defaultunits = [defaultunits]
    sonuc = []

    if not hasattr(convert2pq, "sayilar"):   # make sayilar static variable
        convert2pq.sayilar = deepcopy(sayilar)
        convert2pq.defaultunits=deepcopy(defaultunits)
    elif (convert2pq.sayilar == sayilar) and convert2pq.defaultunits == defaultunits:
        return convert2pq.sonuc

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
        if isinstance(sayi, pq.Quantity):
            sonuc = sayi
            sonuclar.append(float(sonuc.simplified.magnitude))
        elif isinstance(sayi, (float, np.ndarray, int, list)):
            # unit = ""
            # if len(defaultunits) > 0:
                # unit = defaultunits[i]
            # sonuc=pq.Quantity(float(sayi), unit)
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
            if sonuc:
                sonuclar.append(float(sonuc.simplified.magnitude))
            else: # sonuc = None
                print("None input is detected and assumed to be 0!")
                sonuclar.append(0.0)

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

def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    https://matplotlib.org/3.5.0/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Args:
        data: A 2D numpy array of shape (M, N).
        row_labels: A list or array of length M with the labels for the rows.
        col_labels: A list or array of length N with the labels for the columns.
        ax: A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If not provided, use current axes or create a new one.  Optional.
        cbar_kw: A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel: The label for the colorbar.  Optional.
        kwargs: All other arguments are forwarded to `imshow`.

    Usage:
        fig, ax = plt.subplots()
        im, cbar = heatmap(harvest, vegetables, farmers, ax=ax, cmap="YlGn", cbarlabel="harvest [t/year]")
        texts = annotate_heatmap(im, valfmt="{x:.1f} t")

    """
    import matplotlib.pyplot as plt
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw):
    """
    A function to annotate a heatmap.
    https://matplotlib.org/3.5.0/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Args:
    	im: The AxesImage to be labeled.
    	data: Data used to annotate.  If None, the image's data is used.  Optional.
    	valfmt: The format of the annotations inside the heatmap.  This should either use the string format method, e.g. "$ {x:.2f}", or be a `matplotlib.ticker.Formatter`.  Optional.
    	textcolors: A pair of colors.  The first is used for values below a threshold, the second for those above.  Optional.
    	threshold: Value in data units according to which the colors from textcolors are applied.  If None (the default) uses the middle of the colormap as separation.  Optional.
    	kwargs: All other arguments are forwarded to each call to `text` used to create the text labels.

    Usage:
        fig, ax = plt.subplots()
        im, cbar = heatmap(harvest, vegetables, farmers, ax=ax, cmap="YlGn", cbarlabel="harvest [t/year]")
        texts = annotate_heatmap(im, valfmt="{x:.1f} t")

    """
    import matplotlib

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Args:
        x(ndarray): the input signal
        window_len(int, optional): the dimension of the smoothing window; should be an odd integer
        window((string, list, ndarray), optional): either
                    window array with type list or numpy array with size window_len
                or
                    the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'

    Returns:
        the smoothed signal

    Example:
        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)
        see also:
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter

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
    if isinstance(window, (list,np.ndarray)):
        w = np.array(window)
    else:
        if window == 'flat':  #moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]

def str_distance(s, t):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions

    # Computation of the Levenshtein Distance Ratio
    Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
    # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
    # insertions and/or substitutions
    # This is the minimum number of edits needed to convert string a to string b
    # return "The strings are {} edits away".format(distance[rows-1][cols-1])
    return distance[rows-1][cols-1], Ratio

if __name__ == "__main__":
      #rr = polarsample(0.2)
    #print(rr)
    #for r in rr:
    #    print(r)
    #import matplotlib.pyplot as plt
    #import pysmith
    #fig= plt.figure(figsize=(15,10))
    #ax = pysmith.get_smith(fig, 111)
    #for r in rr:
    #    ax.plot([np.real(r)],[np.imag(r)],"*")
    #plt.savefig("Sample_points_02.png")
    #plt.show()

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

    # print(str_distance("rx1","rx2", ratio_calc = False))
    dat = blackman_window(100)
    print(dat)
    pass
