import sys
sys.path.append(r"c:\\users\\erdoel\\documents\\works\\python_works\\microwave_toolbox")
from mwtoolbox.touchstone import *
import network
from constants import *
import itertools
import numpy as np
import matplotlib.pyplot as plt
freq_high=82
freq_low=72
freqs=np.linspace(freq_low*1e9,freq_high*1e9,51)
dL=0.61e-3
out,gamma =thru_line_deembedding("Mean\\Thru_Rx_mean.s2p", "Mean\\Line_Rx_mean.s2p")
freqs=out.get_frequency_list()

Tthru = spfile("Mean\\Thru_Rx_mean.s2p")
Tthru_test = out+out.snp2smp([2,1],0)


plt.figure()
plt.plot(Tthru.get_frequency_list(),Tthru.S(2,1,"db"),label="THRU Meas")
plt.plot(Tthru.get_frequency_list(),Tthru_test.S(2,1,"db"),"*",label="Back2Back Launcher")
plt.grid()
plt.xlabel("Frequency (GHz)")
plt.ylabel("dB")
plt.title("$S_{21}$")
plt.legend()
plt.savefig("TL_S21 dB.png")

plt.figure()
plt.plot(Tthru.get_frequency_list(),Tthru.S(1,1,"db"),label="THRU Meas")
plt.plot(Tthru.get_frequency_list(),Tthru_test.S(1,1,"db"),"*",label="Back2Back Launcher")
plt.grid()
plt.xlabel("Frequency (GHz)")
plt.ylabel("dB")
plt.title("$S_{11}$")
plt.legend()
plt.savefig("TL_S11 dB.png")

plt.figure()
plt.plot(Tthru.get_frequency_list(),Tthru.S(2,1,"phase"),label="THRU Meas")
plt.plot(Tthru.get_frequency_list(),Tthru_test.S(2,1,"phase"),"*",label="Back2Back Launcher")
plt.grid()
plt.xlabel("Frequency (GHz)")
plt.ylabel("Deg")
plt.title("$S_{21}$")
plt.legend()
plt.savefig("TL_S21 phase.png")

plt.figure()
plt.plot(Tthru.get_frequency_list(),Tthru.S(1,1,"phase"),label="THRU Meas")
plt.plot(Tthru.get_frequency_list(),Tthru_test.S(1,1,"phase"),"*",label="Back2Back Launcher")
plt.grid()
plt.xlabel("Frequency (GHz)")
plt.ylabel("Deg")
plt.title("$S_{11}$")
plt.legend()
plt.savefig("TL_S11 phase.png")

# plt.plot(Tthru.S(1,1,"db"),label="thru")
# plt.plot(Tthru_test.S(1,1,"db"),label="test")
# plt.legend()
plt.show()
