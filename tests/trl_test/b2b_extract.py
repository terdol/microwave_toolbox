import sys
sys.path.append(r"c:\\users\\erdoel\\documents\\works\\python_works\\microwave_toolbox")
from touchstone import *
import network
from constants import *
import itertools
import numpy as np
import matplotlib.pyplot as plt
freq_high=82
freq_low=72
freqs=np.linspace(freq_low*1e9,freq_high*1e9,51)
dL=0.61e-3
out,gamma=thru_line_deembedding("WR12_SIW_MS_Transition_1_Design1_Al_SR_Simple1_er307_back2back_unnormalized_1.s2p", "WR12_SIW_MS_Transition_1_Design1_Al_SR_Simple1_er307_back2back_unnormalized_2.s2p")
freqs=out.get_frequency_list()

thru = spfile("WR12_SIW_MS_Transition_1_Design1_Al_SR_Simple1_er307_back2back_unnormalized_1.s2p")

Reference = spfile("WR12_SIW_MS_Transition_1_Design1_Al_SR_Simple1_er307_withadapter_unnormalized_short_largeport.s2p")
print(Reference.refimpedance)
# Reference.change_ref_impedance(50.0)
# Reference=Reference.snp2smp([2,1],0)+Reference
Reference=Reference+Reference.snp2smp([2,1],0)
# out=thru.change_ref_impedance(50.0)
out = out+out.snp2smp([2,1],0)


plt.figure()
plt.plot(Reference.get_frequency_list(),Reference.S(2,1,"db"),label="Reference")
plt.plot(thru.get_frequency_list(),thru.S(2,1,"db"),label="thru")
plt.plot(out.get_frequency_list(),out.S(2,1,"db"),"*",label="Extracted")
plt.grid()
plt.xlabel("Frequency (GHz)")
plt.ylabel("dB")
plt.title("$S_{21}$")
plt.legend()
plt.savefig("TL_S21_dB.png")

plt.figure()
plt.plot(Reference.get_frequency_list(),Reference.S(1,1,"db"),label="Reference")
plt.plot(thru.get_frequency_list(),thru.S(1,1,"db"),label="thru")
plt.plot(out.get_frequency_list(),out.S(1,1,"db"),"*",label="Extracted")
plt.grid()
plt.xlabel("Frequency (GHz)")
plt.ylabel("dB")
plt.title("$S_{11}$")
plt.legend()
plt.savefig("TL_S11_dB.png")

plt.figure()
plt.plot(Reference.get_frequency_list(),Reference.S(2,2,"db"),label="Reference")
plt.plot(thru.get_frequency_list(),thru.S(2,2,"db"),label="thru")
plt.plot(out.get_frequency_list(),out.S(2,2,"db"),"*",label="Extracted")
plt.grid()
plt.xlabel("Frequency (GHz)")
plt.ylabel("Deg")
plt.title("$S_{22}$")
plt.legend()
plt.savefig("TL_S22_phase.png")

# plt.plot(Tthru.S(1,1,"db"),label="thru")
# plt.plot(Tthru_test.S(1,1,"db"),label="test")
# plt.legend()
plt.show()