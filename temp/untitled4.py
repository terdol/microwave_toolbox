# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 11:13:47 2021

@author: erdoel
"""


from touchstone import *

spf = spfile("Chip_impedance_average_allboards_allports.s1p")
spfsmoothed = spf.smoothing(smoothing_length=5,inplace=0)
spfsmoothed.write2file(newfilename="Chip_impedance_average_allboards_allports_smoothed.s1p",parameter="S",freq_unit="GHz",dataformat="RI")
spfsmoothed = spfile("Chip_impedance_average_allboards_allports_smoothed.s1p")
import matplotlib.pyplot as plt

plt.plot(spf.get_frequency_list()/1e9,spf.S(1,1,"dB"),label="Original")
plt.plot(spfsmoothed.get_frequency_list()/1e9,spfsmoothed.S(1,1,"dB"),label="Smooth")
plt.grid()
plt.ylabel("S(1,1)")
plt.legend()
plt.show()
