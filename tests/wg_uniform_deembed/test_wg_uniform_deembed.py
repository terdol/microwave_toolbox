import sys
sys.path.append(r"c:\\users\\erdoel\\documents\\works\\python_works\\microwave_toolbox")
from mwtoolbox.touchstone import *
import itertools
import numpy as np
from mwtoolbox.components import Z_WG_TE10
freq_high=83
freq_low=74
frequencies=np.linspace(freq_low*1e9,freq_high*1e9,10,endpoint=True)
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.autolayout"]=True

rx1in= 1
rx1out= 2
rx2in= 3
rx2out= 4
tx1in= 5
tx1out= 6
tx2in= 7
tx2out= 8

ref = spfile("platedesign_afip_wr10_HFSSDesign6.s8p").uniform_deembed(-1e-3, ports=[1,3,5,7], kind="len")
ref1 = spfile("platedesign_afip_wr10_HFSSDesign6_Zpv_rxin_n1mm_deembed.s8p")

ff="phase"
fig, ax= plt.subplots(2,4,figsize=(15,4))
ax[0,0].plot(ref.freqs/1e9,ref.S(1,1,ff),label="ref")
ax[0,0].plot(ref1.freqs/1e9,ref1.S(1,1,ff),label="ref1")
ax[0,1].plot(ref.freqs/1e9,ref.S(2,2,ff),label="ref")
ax[0,1].plot(ref1.freqs/1e9,ref1.S(2,2,ff),label="ref1")
ax[0,2].plot(ref.freqs/1e9,ref.S(3,3,ff),label="ref")
ax[0,2].plot(ref1.freqs/1e9,ref1.S(3,3,ff),label="ref1")
ax[0,3].plot(ref.freqs/1e9,ref.S(4,4,ff),label="ref")
ax[0,3].plot(ref1.freqs/1e9,ref1.S(4,4,ff),label="ref1")
ax[1,0].plot(ref.freqs/1e9,ref.S(5,5,ff),label="ref")
ax[1,0].plot(ref1.freqs/1e9,ref1.S(5,5,ff),label="ref1")
ax[1,1].plot(ref.freqs/1e9,ref.S(6,6,ff),label="ref")
ax[1,1].plot(ref1.freqs/1e9,ref1.S(6,6,ff),label="ref1")
ax[1,2].plot(ref.freqs/1e9,ref.S(7,7,ff),label="ref")
ax[1,2].plot(ref1.freqs/1e9,ref1.S(7,7,ff),label="ref1")
ax[1,3].plot(ref.freqs/1e9,ref.S(8,8,ff),label="ref")
ax[1,3].plot(ref1.freqs/1e9,ref1.S(8,8,ff),label="ref1")
for aa in fig.axes:
    aa.legend()
    aa.grid()
    aa.set_xlim([freq_low,freq_high])
    aa.set_xlabel("Frequency (GHz)")
    aa.set_ylabel(ff)
# ax[0].set_title("Insertion Loss")
# ax[1].set_title("Input Return Loss")
# ax[2].set_title("Output Return Loss")
# ax[0].set_ylim([-6,3])
# ax[1].set_ylim([-30,0])
# ax[2].set_ylim([-30,0])
plt.savefig(f"deembed_comparison_{ff}.png")
# plt.show()
