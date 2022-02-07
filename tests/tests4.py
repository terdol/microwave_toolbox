import sys
sys.path.append(r"c:\\users\\erdoel\\documents\\works\\python_works\\microwave_toolbox")
from touchstone import *
import network
import itertools
import numpy as np
frequencies=np.linspace(74e9,83e9,8,endpoint=True)

sp = spfile("WR12_SIW_MS_Transition_Design1_29.s2p")
# sp = spfile("WR12_SIW_MS_Transition_Design1_Al_dB_unnormalized.s2p")
# sp = spfile("WR12_SIW_MS_Transition_Design1_Al_RI_unnormalized.s2p")
print(sp.params)
# print(sp.gammas)
sp.set_frequency_points(frequencies)

import matplotlib.pyplot as plt

# plt.plot(frequencies,sp.S(1,1,"dB"))
# plt.plot(frequencies,sp.S(2,2,"dB"))
# plt.grid()

# plt.figure()
# plt.plot(frequencies,sp.S(2,1,"dB"))
# plt.grid()
# plt.show()

# plt.plot(frequencies,np.real(sp.gammas[0]),"+")
# frequencies=np.linspace(74e9,83e9,9,endpoint=True)
# sp.set_frequency_points(frequencies)
# plt.plot(frequencies,np.real(sp.gammas[0]),"*")
# plt.grid()
# plt.show()