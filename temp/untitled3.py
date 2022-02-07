# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 10:37:18 2021

@author: erdoel
"""


import touchstone
import numpy as np
spp = touchstone.spfile("CoaxWG_HFSSDesign2.s2p")
L, Z = touchstone.extractRLGC(spp, 0.01)
print(Z)
import matplotlib.pyplot as plt
plt.plot(spp.get_frequency_list(),np.real(Z))
# plt.plot(spp.get_frequency_list(),np.real(L)*1e9)
plt.ylim([-10,200])
plt.show()