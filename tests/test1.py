import sys
sys.path.append(r"c:\\users\\erdoel\\documents\\works\\python_works\\microwave_toolbox")
from mwtoolbox.touchstone import *
import network
import itertools
import numpy as np
frequencies=np.linspace(57e9,63.5e9,8,endpoint=True)
print(frequencies)
sptline=spfile(noktasayisi=len(frequencies),portsayisi=2)
theta=90
sptline.set_sparam_gen_func(lambda x: network.abcd2s(network.tline(100.0,x*theta/20e9*np.pi/180.0),50.0))
sptline.set_frequency_points(frequencies)

import matplotlib.pyplot as plt
plt.plot(frequencies,sptline.S(1,1,"DB"))
plt.show()
# sptline.set_sparam_gen_func(None)
frequencies=np.linspace(1e9,63.5e9,80,endpoint=True)
sptline.set_frequency_points(frequencies)
import matplotlib.pyplot as plt
plt.plot(frequencies,sptline.S(1,1,"DB"))
plt.show()
