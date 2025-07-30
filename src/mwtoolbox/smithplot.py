# TODO: add annotaiton like:
# ax2.annotate(f"{fhigh_hl}GHz", xy=(np.real(data[-1]), np.imag(data[-1])), xytext=(-100,0), textcoords="offset pixels", arrowprops=dict(facecolor='black', shrink=0.05))
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def get_smith(fig, rect=111, plot_impedance=True, plot_ticks=False, plot_admittance=True, plot_labels=True,
              resticks=None, reactticks=None, xlim=(-1,1), ylim=(-1,1)):

    # font definition
    font = {'family': 'sans-serif',
            'color':  'black',
            'weight': 'normal',
            'size': 18,
            }
    if isinstance(rect, int):
        ax = fig.add_subplot(rect)
    else: #3-tuple
        ax = fig.add_subplot(*rect)
    ax.spines[["bottom"]].set_position("center")
    ax.spines[["left","top","right"]].set_visible(False)
    ax.set_xlim(list(xlim))
    ax.set_ylim(list(ylim))
    ax.set_yticks([])
    ax.spines['bottom'].set_position('zero')

    if plot_impedance:
    
        # make lines of constant resistance
        if resticks is None:
            res_log1 = np.linspace(-4,4,9)
            res1 = 2**res_log1
        else:
            res1 = np.asarray(resticks[:])
        react_log1 = np.linspace(-5,5,2001)
        react = 10**react_log1
        react2 = np.append(-1.0*react[::-1],np.array([0]))
        react = np.append(react2,react)
        for r in res1:
            z = 1j*react + r
            gam = (z-1)/(z+1)
            x = np.real(gam)
            y = np.imag(gam)
            if abs(r-1) > 1e-6:
                ax.plot(x,y,'-k',linewidth = 0.5,alpha=0.5)
            else:
                ax.plot(x,y,'-k',linewidth = 1.0,alpha=1.0)

        # make lines of constant reactance
        if reactticks is None:
            react_log1 = np.linspace(-3,3,7)
            react = 2**react_log1
        else:
            react = np.asarray(reactticks[:])
        res_log1 = np.linspace(-5,5,2001)
        res = 10**res_log1
        react = np.append(-1.0*react[::-1],react)
        for chi in react:
            z = 1j*chi + res
            gam = (z-1)/(z+1)
            x = np.real(gam)
            y = np.imag(gam)
            if abs(chi-1) > 1e-6 and abs(chi+1) > 1e-6 and abs(chi) > 1e-6:
                ax.plot(x,y,'-k',linewidth = 0.5,alpha=0.5)
            else:
                ax.plot(x,y,'-k',linewidth = 1.0,alpha=1.0)
            if min(x)>xlim[1] or max(x)<xlim[0] or min(y)>ylim[1] or max(y)<ylim[0]:
                continue
            x1, x2 = max(xlim[0], x[0]), min(xlim[1], x[-1])
            y1, y2 = y[np.argmin(np.abs(x-x1))], y[np.argmin(np.abs(x-x2))]
            y1, y2 = max(ylim[0], min(y1,y2)), min(ylim[1], max(y1,y2))
            x1, x2 = x[np.argmin(np.abs(y-y1))], x[np.argmin(np.abs(y-y2))]
            xo, yo = (x1, y1) if x1<x2 else (x2, y2)
            if plot_labels:
                text = ax.text(xo, yo, f"{chi:.02f}", fontsize=font["size"]/3)

    if plot_admittance:

        # make lines of constant conductance
        if resticks is None:
            res_log1 = np.linspace(-4,4,9)
            res1 = 2**res_log1
        else:
            res1 = np.asarray(resticks[:])
        react_log1 = np.linspace(-5,5,2001)
        react = 10**react_log1
        react = np.append(-1.0*react[::-1],react)
        for r in res1:
            y = 1.0/r + 1.0/(1j*react)
            gam = (1.0/y-1)/(1.0/y+1)
            x = np.real(gam)
            y = np.imag(gam)
            if abs(r-1) > 1e-6:
                ax.plot(x,y,'-k',linewidth = 0.5,alpha=0.5)
            else:
                ax.plot(x,y,'-k',linewidth = 1.0,alpha=1.0)

        # make lines of constant susceptance
        if reactticks is None:
            react_log1 = np.linspace(-3,3,7)
            react = 2**react_log1
        else:
            react = np.asarray(reactticks[:])
        res_log1 = np.linspace(-5,5,2001)
        res = 10**res_log1
        react = np.append(-1.0*react[::-1],react)
        for chi in react:
            y = 1.0/(1j*chi) + 1.0/res
            gam = (1.0/y-1)/(1.0/y+1)
            x = np.real(gam)
            y = np.imag(gam)
            if abs(chi-1) > 1e-6 and abs(chi+1) > 1e-6 and abs(chi) > 1e-6:
                ax.plot(x,y,'-k',linewidth = 0.5,alpha=0.5)
            else:
                ax.plot(x,y,'-k',linewidth = 1.0,alpha=1.0)
            if min(x)>xlim[1] or max(x)<xlim[0] or min(y)>ylim[1] or max(y)<ylim[0]:
                continue
            x1, x2 = max(xlim[0], x[0]), min(xlim[1], x[-1])
            y1, y2 = y[np.argmin(np.abs(x-x1))], y[np.argmin(np.abs(x-x2))]
            y1, y2 = max(ylim[0], min(y1,y2)), min(ylim[1], max(y1,y2))
            x1, x2 = x[np.argmin(np.abs(y-y1))], x[np.argmin(np.abs(y-y2))]
            xo, yo = (x1, y1) if x1>x2 else (x2, y2)
            if plot_labels:
                text = ax.text(xo, yo, f"{chi:.02f}", fontsize=font["size"]/3)


    theta = np.linspace(-np.pi, np.pi, 10000)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.plot(x,y,'-k',linewidth=0.5, alpha=0.5)

    if plot_labels:
        xt = np.array([xa for xa in (res1-1)/(res1+1) if xlim[1]>xa>xlim[0]])
        res1 = (1+xt)/(1-xt)
        ax.set_xticks(xt)
        ax.set_xticklabels([f"{rr:.2f}" if rr<1000 else "$\infty$" for rr in res1], fontsize = font["size"]/3)

    ax.set_aspect("equal")
    return ax


if __name__ == "__main__":
    
    fig = plt.figure(figsize=(5,5))
    ax = get_smith(fig, 111, plot_ticks=True, plot_admittance=False, plot_labels=True)
    plt.show()
