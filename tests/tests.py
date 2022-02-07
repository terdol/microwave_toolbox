import sys
sys.path.append(r"c:\\users\\erdoel\\documents\\works\\python_works\\microwave_toolbox")
from touchstone import *
import network
import itertools
import numpy as np
frequencies=np.linspace(57e9,60e9,4,endpoint=True)

sptline=spfile(freqs=frequencies,portsayisi=2)
sptline.set_frequency_points(frequencies)
theta=90
for i in range(len(frequencies)):
    sptline.set_smatrix_at_frequency_point(i,network.abcd2s(network.shunt_z(100.0)*network.series_z(100.0),50.0))
# print(sptline.S(1,1))
print(sptline.sdata)
# sptline.change_ref_impedance(100.0)
sptline.change_ref_impedance([100+50j,30-25j])
sptline.change_ref_impedance([100-50j,30+25j])
# sptline.change_ref_impedance([100,30])
print(sptline.sdata)
sptline.calc_syz("S")
print(sptline.zdata)
# print(sptline.S(1,1))