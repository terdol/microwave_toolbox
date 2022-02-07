import sys
sys.path.append(r"c:\\users\\erdoel\\documents\\works\\python_works\\microwave_toolbox")
from touchstone import *
import network
import itertools
import numpy as np
frequencies=np.linspace(57e9,60e9,4,endpoint=True)

sptline=spfile(freqs=[10e9],portsayisi=2)
# sptline.set_frequency_points([10e9])
sptline.set_smatrix_at_frequency_point(0,np.array([[0.2,0.3],[0.1,0.5]]))
sptline.set_frequency_points([20e9,30e9,40e9])
print(sptline.sdata)