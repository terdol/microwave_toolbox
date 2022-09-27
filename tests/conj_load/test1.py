import sys
sys.path.append(r"c:\\users\\erdoel\\documents\\works\\python_works\\microwave_toolbox")
from mwtoolbox.touchstone import *
import network
import itertools
import numpy as np

sp = spfile("HalfModel_CPW5_DEMSB_RX1.s2p")
frequencies = sp.get_frequency_list()
gs, gl = sp.Z_conjmatch()
zs=50.0*(1+gs)/(1-gs)
zl=50.0*(1+gl)/(1-gl)
import matplotlib.pyplot as plt
plt.plot(frequencies,np.real(zs),label="Zs Real")
plt.plot(frequencies,np.real(zl),label="Zl Real")
plt.plot(frequencies,np.imag(zs),label="Zs Imag")
plt.plot(frequencies,np.imag(zl),label="Zl Imag")
plt.legend()
plt.grid()
plt.savefig("impedances.png")


sp.change_ref_impedance([zs,zl])
plt.figure()
plt.plot(frequencies,sp.S(1,1),label="S(1,1)")
plt.plot(frequencies,sp.S(2,2),label="S(2,2)")
plt.grid()
plt.legend()


plt.show()
