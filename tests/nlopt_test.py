import nlopt
import numpy as np
def ff(x,grad=0):
    return sum((0.15*x**4+0.6*x**3-0.3*x**2-0.2*x+0.6-0.565)**2)
# print(ff(0.15))
opt = nlopt.opt(nlopt.GN_ESCH, 1)
opt.set_min_objective(ff)
opt.set_lower_bounds([0])
opt.set_upper_bounds([1])
# opt.set_maxeval(1000)
# opt.set_maxtime(1)
opt.set_ftol_abs(1e-5)
xopt = opt.optimize(np.array([0.8]))
print(xopt)
print(ff(np.array([0.15043442])))
print(ff(np.array([0.72])))