import sys
sys.path.append(r"c:\\users\\erdoel\\documents\\works\\python_works\\microwave_toolbox")
from mwtoolbox.rfnetwork import *
import network
import itertools
import numpy as np

sp = spfile("HalfModel_CPW5_DEMSB_RX1.s2p")
frequencies = sp.get_frequency_list()
sp1 = sp.conj_match_uncoupled2(inplace=0, noofiters=1)
sp2 = sp.conj_match_uncoupled2(inplace=0, noofiters=50)

zs, zl = sp.Z_conjmatch()
zs1=sp1.prepare_ref_impedance_array(sp1.refimpedance)[0]
zs2=sp2.prepare_ref_impedance_array(sp2.refimpedance)[0]
zl1=sp1.prepare_ref_impedance_array(sp1.refimpedance)[1]
zl2=sp2.prepare_ref_impedance_array(sp2.refimpedance)[1]
import matplotlib.pyplot as plt
fig,ax=plt.subplots(2,2)
ax[0,0].plot(frequencies,np.real(zs),label="Zs Real")
ax[0,0].plot(frequencies,np.real(zs1),label="Zs1 Real")
ax[0,0].plot(frequencies,np.real(zs2),"*",label="Zs2 Real")
ax[0,1].plot(frequencies,np.real(zl),label="Zl Real")
ax[0,1].plot(frequencies,np.real(zl1),label="Zl1 Real")
ax[0,1].plot(frequencies,np.real(zl2),"*",label="Zl2 Real")
ax[1,0].plot(frequencies,np.imag(zs),label="Zs Imag")
ax[1,0].plot(frequencies,np.imag(zs1),label="Zs1 Imag")
ax[1,0].plot(frequencies,np.imag(zs2),"*",label="Zs2 Imag")
ax[1,1].plot(frequencies,np.imag(zl),label="Zl Imag")
ax[1,1].plot(frequencies,np.imag(zl1),label="Zl1 Imag")
ax[1,1].plot(frequencies,np.imag(zl2),"*",label="Zl2 Imag")
plt.legend()
plt.grid()
plt.savefig("impedances1.png")

plt.show()
