#-*-coding:utf-8-*-
from genel import *
speed_of_light_in_freespace=299792458*pq.m/pq.s
free_space_wave_impedance=376.730313461*pq.ohm
free_space_permeability=12.566370614e-7*pq.N/pq.A/pq.A
free_space_permittivity=8.854187817e-12*pq.F/pq.m

mu0 = free_space_permeability.simplified.magnitude
eps0 = free_space_permittivity.simplified.magnitude
co = c0 = speed_of_light_in_freespace.simplified.magnitude
