import cmath
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
from mpl_toolkits.axisartist import SubplotHost
from mpl_toolkits.axisartist import GridHelperCurveLinear
import mpl_toolkits.axisartist.floating_axes as floating_axes
from mpl_toolkits.axisartist.grid_finder import (FixedLocator, MaxNLocator,
                                                 DictFormatter)



def get_smith(fig, rect = 111, plot_impedance = True, plot_ticks = False, plot_admittance = True, plot_labels = True):

    # font definition
    font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        'size': 18,
        }

    ax = fig.add_subplot(rect)
    ax.spines[["bottom"]].set_position("center")
    ax.spines[["left","top","right"]].set_visible(False)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_yticks([])
    if plot_impedance:
        # make lines of constant resistance
        res_log = np.linspace(-4,4,9)
        res_log = np.insert(res_log, 0, -20)
        react_log = np.linspace(-5,5,2001)
        res = 2**res_log
        react = 10**react_log
        react2 = np.append(-1.0*react[::-1],np.array([0]))
        react = np.append(react2,react)
        for r in res:
            z = 1j*react + r
            gam = (z-1)/(z+1)
            x = np.real(gam)
            y = np.imag(gam)
            if abs(r-1) > 1e-6:
                ax.plot(x,y,'-k',linewidth = 0.5,alpha=0.5)
            else:
                ax.plot(x,y,'-k',linewidth = 1.0,alpha=1.0)
        if plot_labels:
            print((res-1)/(res+1))
            ax.set_xticks((res-1)/(res+1))
            ax.set_xticklabels([f"{rr:.2f}" for rr in res], fontsize = font["size"]/3)

        # make lines of constant reactance
        react_log = np.linspace(-3,3,7)
        res_log = np.linspace(-5,5,2001)
        res = 10**res_log
        react = 2**react_log
        react2 = np.append(-1.0*react[::-1],np.array([0]))
        react = np.append(react2,react)
        for chi in react:
            z = 1j*chi + res
            gam = (z-1)/(z+1)
            x = np.real(gam)
            y = np.imag(gam)
            if abs(chi-1) > 1e-6 and abs(chi+1) > 1e-6 and abs(chi) > 1e-6:
                ax.plot(x,y,'-k',linewidth = 0.5,alpha=0.5)
            else:
                ax.plot(x,y,'-k',linewidth = 1.0,alpha=1.0)
            pos = (1j*chi-1)/(1j*chi+1)
            if plot_labels:
                text = ax.text(np.real(pos),np.imag(pos),f"{chi:.02f}",fontsize = font["size"]/3)

    if plot_admittance:
        # make lines of constant conductance
        res_log = np.linspace(-4,4,9)
        react_log = np.linspace(-5,5,2001)
        res = 2**res_log
        react = 10**react_log
        react = np.append(-1.0*react[::-1],react)
        for r in res:
            y = 1.0/r + 1.0/(1j*react)
            gam = (1.0/y-1)/(1.0/y+1)
            x = np.real(gam)
            y = np.imag(gam)
            if abs(r-1) > 1e-6:
                ax.plot(x,y,c='k',linewidth = 0.1,alpha=0.5)
            else:
                ax.plot(x,y,c='k',linewidth = 0.1,alpha=0.85)
        # make lines of constant susceptance
        react_log = np.linspace(-3,3,7)
        res_log = np.linspace(-5,5,2001)
        res = 10**res_log
        react = 2**react_log
        react = np.append(-1.0*react[::-1],react)
        for chi in react:
            y = 1.0/(1j*chi) + 1.0/res
            gam = (1.0/y-1)/(1.0/y+1)
            x = np.real(gam)
            y = np.imag(gam)
            if abs(chi-1) > 1e-6 and abs(chi+1) > 1e-6:
                ax.plot(x,y,c='k',linewidth = 0.1,alpha=0.5)
            else:
                ax.plot(x,y,c='k',linewidth = 0.1,alpha=0.85)
        y = 1.0/res
        gam = (1.0/y-1)/(1.0/y+1)
        x = np.real(gam)
        y = np.imag(gam)
        ax.plot(x,y,c='k',linewidth = 0.1,alpha=0.75)

    ax.set_aspect("equal")
    return ax


if __name__ == "__main__":
    
    fig = plt.figure(figsize=(5,5))
    ax = get_smith(fig, 111, plot_ticks=True, plot_admittance=False, plot_labels=True)
    plt.show()
