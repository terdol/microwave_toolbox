import sys
sys.path.append(r"c:\\users\\erdoel\\documents\\works\\python_works\\microwave_toolbox")
from mwtoolbox.rfnetwork import *
import network
import itertools
import numpy as np
freq_high=83
freq_low=74
frequencies=np.linspace(freq_low*1e9,freq_high*1e9,41,endpoint=True)
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.autolayout"]=True
Lsp = []
# mainfilename = "WR12_SIW_MS_Transition_2_Testpcb_u.s8p"
#mainfilename = "WR12_SIW_MS_Transition_2_Testpcb_singlepart1_u.s8p"
mainfilename = "WR12_SIW_MS_Transition_2_Testpcb_singlepart_noplating.s8p"
#mainfilename = "WR12_SIW_MS_Transition_2_Testpcb_singlepart_u.s8p"
version="_singlepart_noplating_"
fullsimsp=spfile(mainfilename)
freks = fullsimsp.get_frequency_list()
L1sp = fullsimsp.snp2smp([2,1],inplace=0)
zref = np.array(L1sp.refimpedance[0])
L1sp_short = fullsimsp.snp2smp([2,1],inplace=0).change_ref_impedance([None,0.01],inplace=0)
L1sp_open = fullsimsp.snp2smp([2,1],inplace=0).change_ref_impedance([None,10000],inplace=0)
L1sp_load = fullsimsp.snp2smp([2,1],inplace=0)

gL1=0.01
gL2=(0.01-zref)/(0.01+zref)
gL3=(10000-zref)/(10000+zref)
g1=L1sp_load.S(1,1)
g2=L1sp_short.S(1,1)
g3=L1sp_open.S(1,1)

output1 = untermination_method(g1,g2,g3,gL1,gL2,gL3,returnS2P=True, freqs=freks)
Lsp.append(output1)
index2 = 4
index1 = 3
L1sp = fullsimsp.snp2smp([index2,index1],inplace=0)
zref = np.array(L1sp.refimpedance[0])
ref = fullsimsp.snp2smp([index2,index1],inplace=0)
L1sp_short = ref.change_ref_impedance([None,0.01],inplace=0)
L1sp_open = ref.change_ref_impedance([None,10000],inplace=0)
L1sp_load = ref

fig, ax= plt.subplots(1,3,figsize=(15,4))
#ax[0].plot(L1sp_short.get_frequency_list()/1e9,L1sp_short.S(1,1,"dB"),label="Short")
#ax[1].plot(L1sp_open.get_frequency_list()/1e9,L1sp_open.S(1,1,"dB"),label="Open")
#ax[2].plot(L1sp_load.get_frequency_list()/1e9,L1sp_load.S(1,1,"dB"),label="Load")
#for aa in fig.axes:
#    aa.legend()
#    aa.grid()
#    aa.set_xlim([freq_low,freq_high])
#    aa.set_xlabel("Frequency (GHz)")
#    aa.set_ylabel("dB")
#ax[0].set_title("Return Loss")
#ax[1].set_title("Return Loss")
#ax[2].set_title("Return Loss")
#ax[0].set_ylim([-30,0])
#ax[1].set_ylim([-30,0])
#ax[2].set_ylim([-30,0])
#plt.savefig("returnloss_short_open_load.png")
#plt.show()

gL1=0.0001
gL2=(0.01-zref)/(0.01+zref)
gL3=(10000-zref)/(10000+zref)
g1=L1sp_load.S(1,1)
g2=L1sp_short.S(1,1)
g3=L1sp_open.S(1,1)

output2 = untermination_method_old(g1,g2,g3,gL1,gL2,gL3,returnS2P=True, freqs=freks)
output3 = untermination_method(g1,g2,g3,gL1,gL2,gL3,returnS2P=True, freqs=freks)
Lsp.append(output2)

ax[0].plot(output2.get_frequency_list()/1e9,output2.S(2,1,"dB"),label="Old")
ax[0].plot(output3.get_frequency_list()/1e9,output3.S(2,1,"dB"),"+",label="New")
ax[0].plot(ref.get_frequency_list()/1e9,ref.S(2,1,"dB"),"*",label="Ref")
ax[1].plot(output2.get_frequency_list()/1e9,output2.S(1,1,"dB"),label="Old")
ax[1].plot(output3.get_frequency_list()/1e9,output3.S(1,1,"dB"),"+",label="New")
ax[1].plot(ref.get_frequency_list()/1e9,ref.S(1,1,"dB"),"*",label="Ref")
ax[2].plot(output2.get_frequency_list()/1e9,output2.S(2,2,"dB"),label="Old")
ax[2].plot(output3.get_frequency_list()/1e9,output3.S(2,2,"dB"),"+",label="New")
ax[2].plot(ref.get_frequency_list()/1e9,ref.S(2,2,"dB"),"*",label="Ref")
for aa in fig.axes:
    aa.legend()
    aa.grid()
    aa.set_xlim([freq_low,freq_high])
    aa.set_xlabel("Frequency (GHz)")
    aa.set_ylabel("dB")
ax[0].set_title("Insertion Loss")
ax[1].set_title("WG Return Loss")
ax[2].set_title("MS Return Loss")
ax[0].set_ylim([-6,0])
ax[1].set_ylim([-30,0])
ax[2].set_ylim([-30,0])
plt.savefig("single_transition_unterm.png")
plt.show()

#fig, ax= plt.subplots(2,3,figsize=(15,8))
#ax[0,0].plot(L1sp.get_frequency_list()/1e9,output2.S(2,1,"dB"),label="Extracted")
#ax[1,0].plot(L1sp.get_frequency_list()/1e9,L1sp.S(2,1,"dB"),label="Original")
#ax[0,1].plot(L1sp.get_frequency_list()/1e9,output2.S(1,1,"dB"),label="Extracted")
#ax[1,1].plot(L1sp.get_frequency_list()/1e9,L1sp.S(1,1,"dB"),label="Original")
#ax[0,2].plot(L1sp.get_frequency_list()/1e9,output2.S(2,2,"dB"),label="Extracted")
#ax[1,2].plot(L1sp.get_frequency_list()/1e9,L1sp.S(2,2,"dB"),label="Original")
#for aa in fig.axes:
#    aa.legend()
#    aa.grid()
#    aa.set_xlim([freq_low,freq_high])
#    aa.set_xlabel("Frequency (GHz)")
#    aa.set_ylabel("dB")
#ax[0,0].set_title("Insertion Loss")
#ax[0,1].set_title("WG Return Loss")
#ax[0,2].set_title("MS Return Loss")
#ax[0,0].set_ylim([-7,0])
#ax[0,1].set_ylim([-30,0])
#ax[0,2].set_ylim([-30,0])
#ax[1,0].set_ylim([-7,0])
#ax[1,1].set_ylim([-30,0])
#ax[1,2].set_ylim([-30,0])
#plt.savefig("unterm.png")
#plt.show()

#fig, ax= plt.subplots(1,3,figsize=(15,4))
#ax[0].plot(output.get_frequency_list()/1e9,output.S(2,1,"dB"),label="On Board")
#ax[1].plot(output.get_frequency_list()/1e9,output.S(1,1,"dB"),label="On Board")
#ax[2].plot(output.get_frequency_list()/1e9,output.S(2,2,"dB"),label="On Board")
#for aa in fig.axes:
#    aa.legend()
#    aa.grid()
#    aa.set_xlim([freq_low,freq_high])
#    aa.set_xlabel("Frequency (GHz)")
#    aa.set_ylabel("dB")
#ax[0].set_title("Insertion Loss")
#ax[1].set_title("WG Return Loss")
#ax[2].set_title("MS Return Loss")
#ax[0].set_ylim([-6,0])
#ax[1].set_ylim([-30,0])
#ax[2].set_ylim([-30,0])
#plt.savefig("transition"+version+".png")
