# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython


# %%
from numpy import *
w=3.1e-3
h=1.55e-3
t=0.9e-3
e0=8.85e-12
eps=3.4
fr=77e9
kc=pi/w
co=3e8
k=2*pi*fr*sqrt(eps)/co
beta=sqrt(k**2-kc**2)
lg=2*pi/beta
print(str(lg/4))


# %%
import quantities as pq
dd=pq.Quantity("100um")
print(str(dd.simplified.magnitude))


# %%
get_ipython().run_line_magic('matplotlib', 'widget')
import matplotlib.pyplot as plt
plt.plot([1,5,4])
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
fig, ax = plt.subplots(figsize=(8, 5))

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = ax.plot(t, s, lw=3)
el = Ellipse((2, np.cos(2*np.pi*2)), 0.1, 0.2)
ax.add_patch(el)
ann = ax.annotate('bubble',
                  xy=(2.,np.cos(2*np.pi*2)), xycoords='data',
                  xytext=(0, 55), textcoords='offset points',
                  size=20, va="center",
                  bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"),
                  arrowprops=dict(arrowstyle="wedge,tail_width=1.",
                                  fc=(1.0, 0.7, 0.7), ec="none",alpha=0.2,
                                  patchA=None,
                                  patchB=None,
                                  relpos=(0.2, 0.5)))
ann.draggable()
ax.set(xlim=(-1, 5), ylim=(-5, 3))

# %%
a=[1,2,3,4,5,6,7,8,9]
b=[x for x in range(len(a)) if a[x]>3 and a[x]<8]
print(b)
a[tuple(b)]


# %%
import os, sys
os.chdir(r"C:\\Users\\Erdoel\\Documents\\Works\\AFiP\\Simulations\\Analiz\\Foam")
sys.path.append(r"C:\\Users\\Erdoel\\Documents\\Works\\Python_Works\\microwave_toolbox")
from touchstone import spfile
sp = spfile("Transition1_Foam_tolerance_Isolation_Rect_xo_0um_yo_400um_zo_0um_unnormalized.s4p")
print(sp.params)


# %%
groupusers=["abaei","engelsbe","kornpro","doganmus", "wojnows", "arifro", "erdoel", "qureshi", "schittl","horedt","frances","napieral"]
user="CEngelsbe"
dd=filter(lambda x:x in user.lower(), groupusers)
print(list(dd))

# %%
import numpy as np
aa= np.array([1,2,3,4,6])
print(f"{aa[2]:.2f}")
print(3e8/77e9/np.sqrt(3.76))

# %%
from scipy.special import *
print(jnp_zeros(1,3))
print(jn_zeros(1,3))

# %%
from pytexit import py2tex
py2tex("1.0/pi*(s*log((q/a))+(1.0-s)*log(((1.0-q)/(1.0+a))))")

# %%
from scipy.special import ellipk
import numpy as np
x=np.linspace(0.05,0.95,91)
coefs = np.polyfit(x,ellipk(x),9)
p=np.poly1d(coefs)
import matplotlib.pyplot as plt
plt.plot(x,ellipk(x))
plt.plot(x,p(x),"r")
plt.show()

# %%
import numpy as np
a=[np.array([5,6]),np.array([7,8]),np.array([9,10]),np.array([11,12])]
b=np.array(a).reshape((1,8)).reshape((2,2,2))
#b=np.matrix([[1,np.array([5,6])],[3,4]])
#print(b)
np.conj(3+4j)

# %%
import numpy as np
import sys
sys.path.append(r"c:\\users\\erdoel\\documents\\works\\python_works\\microwave_toolbox")
import matplotlib.pyplot as plt
from scipy.signal import *
from genel import *
x=blackman(101)
print(x[48:52])
plt.plot(blackman(101))
plt.show()


# %%
import sympy
a1,b1,c1,d1,a2,b2,c2,d2 = sympy.symbols(["a1","b1","c1","d1","a2","b2","c2","d2"])
M1=sympy.Matrix([[a1,b1],[c1,d1]])
M2=sympy.Matrix([[a2,b2],[c2,d2]])
M = M1*M2
# sympy.pprint(M)
# print(sympy.I*sympy.I)
print(M1.shape[0])

# %% [markdown]
# $$\int_{0}^{\infty}x^2dx$$

# %%
a=5
print(f"{a*3:.0f}")
print("TnZ".lower())


# %%
s11=1.8926848471E-001 +2.8106090426E-001*1j
import numpy as np
print(20*np.log10(np.abs(s11)))
lam = 3e8/77e9/np.sqrt(4.07)
120e-6/lam*360


# %%
import numpy as np
b=np.matrix(np.diag([2,3,4,5,6,7,8]))
ixgrid=np.ix_([2,3,4], [2,3,4])
print(b[ixgrid])


# %%
import itertools
list1 = [True, False]
list2 = [0, 1, 2]
all_combinations = []

list2_permutations = itertools.permutations(list2, len(list1))
for each_permutation in list2_permutations:
    zipped = zip(each_permutation, list1)
    all_combinations.append(list(zipped))

print(all_combinations)


# %%
class temp():
    pass
aa=temp()
aa.b=3
aa.__dict__["c"]=34
print(aa.__dict__)


# %%
from tkinter import *

m1 = PanedWindow()
m1.pack(fill=BOTH, expand=1)

left = Label(m1, text="left pane")
m1.add(left)

m2 = PanedWindow(m1, orient=VERTICAL)
m1.add(m2)

top = Label(m2, text="top pane")
m2.add(top)

bottom = Label(m2, text="bottom pane")
m2.add(bottom)

mainloop()


# %%
import numpy as np
aa = np.array([1.1,2.1,3.1])
aa**1.2


# %%
import numpy as np
a=np.array(list(range(10,20)))
print(a[[2,5,8]])


# %%
import matplotlib.pyplot as plt
plt.ion()
plt.plot([1,2,3])
plt.arrow(2,2,-1,0, head_width=0.05, head_length=0.1, fc='k', ec='k')
plt.show()


# %%
# TE-case
from numpy import *
from sympy import *
init_printing()
w,u,a,b,kz,kc,x,y=symbols("w u a b k_z k_c x y")
n = Symbol("n",integer=True)
m = Symbol("m",integer=True, positive=True)
Q.is_true(m>0)
Q.is_true(a>0)
Q.is_true(b>0)
Q.is_true(kc>0)
Q.is_true(kz>0)
Q.is_true(w>0)
Q.is_true(u>0)
def ex(x):
    return 1j*w*u/kc**2*(n*pi/b)*cos(m*pi*x/a)*sin(n*pi*y/b)
def ey(x):
    return -1j*w*u/kc**2*(m*pi/a)*sin(m*pi*x/a)*cos(n*pi*y/b)
def hx(x):
    return 1j*kz/kc**2*(m*pi/a)*sin(m*pi*x/a)*cos(n*pi*y/b)
def hy(x):
    return 1j*kz/kc**2*(n*pi/b)*cos(m*pi*x/a)*sin(n*pi*y/b)
aa = integrate(ex(x)*by(x)-ey(x)*hx(x),(x,0,a),(y,0,b))
display(aa)


# %%
a={5:"a",2:"b",3:"c"}
bb=list(a.values())
print(bb.index("c"))


# %%
a=(1,2,3)
b=a+(5,)
print(b)


# %%
h=[1,2,3]
h.pop(h.index(3))
print(h)


# %%
import math
a=math.sqrt(0.9657072277**2+0.12797206**2)
10*log10(a**2)


# %%
val=[1,2,3]
it=["a","b","c"]
print(dict(zip(it,val)))


# %%
import numpy as np
aa = np.matrix([[1+2j,3+5j],[2+1j,1+5j]])
np.linspace(0.1,0.9,9)
print(3e8/3.1e-3/np.sqrt(3))


# %%

