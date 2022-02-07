
import csv
import difflib
# from numpy.lib.scimath import sqrt as csqrt
from constants import *
import numpy as np

def skin_depth(f, material):
    """
    This function returns skin depth in meters.
    Argument List:
    First 3 arguments are inputs.
    1- f ; frequency
    2- material ; string
    """
    MatDict={}
    rdr=csv.reader(open("MaterialProperties.csv"),delimiter=";")
    next(rdr)
    for row in rdr:
        MatDict[row[0]]=[float(x.replace(",",".")) for x in row[1:-1]]+[int(row[-1])]
    x=difflib.get_close_matches(material,MatDict.keys(), cutoff=0.2)
    sigma = MatDict[x[0]][4]
    mu = MatDict[x[0]][2]
    skindepth = np.sqrt(1.0 / sigma / mu / mu0 / np.pi / f)
    print("Material: "+x[0])
    print("Skin Depth: "+str(skindepth*1e6)+" um")
    return skindepth