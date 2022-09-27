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
# board="Board04"
# Tlauncherout,Tlauncherin,faz=trl_launcher_extraction(board+"_Thru_Rx.s2p", board+"_Line_Rx.s2p", board+"_Short_Rx.s2p", refstd=False)
Tlauncherout,Tlauncherin,faz=trl_launcher_extraction("Mean\\Thru_Rx_mean.s2p", "Mean\\Line_Rx_mean.s2p", "Mean\\Short_Rx_mean.s2p", refstd=False)
# Tlauncher.set_port_names(["PROBE","PKG"])
# Tlauncher.write2file(newfilename=board+"_Launcher",parameter="S",freq_unit="GHz",format="DB")
# Tlauncher.write2file(newfilename="Mean\\Launcher",parameter="S",freq_unit="GHz",format="DB")
# Tthru = spfile(board+"_Thru_Rx.s2p")
Tthru = spfile("Mean\\Thru_Rx_mean.s2p")
# Tline = spfile(board+"_Line_Rx.s2p")

# Tthru_test = Tlauncherin+Tlauncherout.snp2smp([2,1],0)
Tthru_test = Tlauncherout+Tlauncherout.snp2smp([2,1],0)

# freqs=Tlauncher.get_frequency_list()
# eeff=(np.imag(faz)*c0/(2*np.pi*freqs)/dL)**2
# atten=-20*np.real(faz)*np.log10(np.exp(1))/dL/1000

# plt.plot(Tthru.S(2,1,"phase"),label="thru")
# plt.plot(freqs, eeff,label="test")
# plt.plot(freqs, atten,label="test")
# plt.legend()

# plt.plot(Tthru.S(2,1,"phase"),label="thru")
# plt.plot(Tthru_test.S(2,1,"phase"),label="test")
# plt.legend()

plt.figure()
plt.plot(Tthru.get_frequency_list(),Tthru.S(2,1,"db"),label="THRU Meas")
plt.plot(Tthru.get_frequency_list(),Tthru_test.S(2,1,"db"),"*",label="Back2Back Launcher")
plt.grid()
plt.xlabel("Frequency (GHz)")
plt.ylabel("dB")
plt.title("$S_{21}$")
plt.legend()
plt.savefig("S21 dB.png")

plt.figure()
plt.plot(Tthru.get_frequency_list(),Tthru.S(1,1,"db"),label="THRU Meas")
plt.plot(Tthru.get_frequency_list(),Tthru_test.S(1,1,"db"),"*",label="Back2Back Launcher")
plt.grid()
plt.xlabel("Frequency (GHz)")
plt.ylabel("dB")
plt.title("$S_{11}$")
plt.legend()
plt.savefig("S11 dB.png")

plt.figure()
plt.plot(Tthru.get_frequency_list(),Tthru.S(2,1,"phase"),label="THRU Meas")
plt.plot(Tthru.get_frequency_list(),Tthru_test.S(2,1,"phase"),"*",label="Back2Back Launcher")
plt.grid()
plt.xlabel("Frequency (GHz)")
plt.ylabel("Deg")
plt.title("$S_{21}$")
plt.legend()
plt.savefig("S11 phase.png")

# plt.plot(Tthru.S(1,1,"db"),label="thru")
# plt.plot(Tthru_test.S(1,1,"db"),label="test")
# plt.legend()
plt.show()
