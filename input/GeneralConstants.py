'''
---------- EXPLANATION ----------
In these constants the unit is included and the definitiveness
(if that's a word) of the variable.
So an O means that this value can be taken as true, 
so for example the atmospheric constants won't change 
and the cruise speed has already been defined by requirements. 
An ~ means it's defined but maybe not needed and a ? means it's a guess.


'''
import numpy as np
import sys
import pathlib as pl
import os

sys.path.append(str(list(pl.Path(__file__).parents)[1]))

from  modules.misc_tools.tools import ISA

#performance
v_cr = 300/3.6
h_cruise = 400          #[m]        O
mission_dist = 400e3  # km
npax = 4                # Amount of passengers O

#atmospheric constants
atm = ISA(h_cruise)
g0 = 9.80665            #[m/s^2]    O
rho_cr = atm.density()    #[kg/m^3]   O
p_cr = atm.pressure()           # pressure at cr O
t_cr = atm.temperature()      # Temperature at cr O
a_cr = atm.soundspeed()     #Speed of sound at cruise O
R = 287                 #[J/kg*K]   O

# Sea leavel atmospheric constants
rho_sl = atm.p_SL            #[kg/m^3]   O
p_sl = atm.rho_SL
T_sl = atm.T_SL
mhu_sl = atm.mu_SL
a_sl = atm.a_SL
mhu = atm.viscosity_dyn()           #[kg/m^3]   O   the dynamic viscosity


# Power
p_density = 3.117e3     # w totalEnergy/kg    ? # averaged from:  A review: high power density motors for electric vehicles
DOD = 0.8
ChargingEfficiency = 0.7

#standard masses
m_pl = 475 # kg


#airfoil
toc = 0.169 #NACA44017
xcm = 0.293 # NACA44017
A_base = 0 #Assumed 0 base area
frac_lam_fus = 0.05
frac_lam_wing = 0.1
k = 0.634 * 10**(-5) # Surface smoothness parameter

#fuelcell input
VolumeDensityFuellCell = 3.25 #kW /l
PowerDensityFuellCell = 3.9 #kW/kg
effiencyFuellCell = 0.6

#Tank input
VolumeDensityTank = 0.5 #kg/l
EnergyDensityTank = 1.85 # kWh/kg

# Requirements
n_min = -1              # [-]       O   Min load factor
n_max = 2.5             # [-]       O   Max load factor
n_ult = 1.5*n_max       # [-]       O   Ultimate load factor
ub = 20.12              # [m/s]     0   Gust at Vb from EASA
uc = 15.24              # [m/s]     0   Gust at Vc from EASA
ud = 7.62               # [m/s]     0   Gust at Vd from EASA
