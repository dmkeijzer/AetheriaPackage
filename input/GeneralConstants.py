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

from  modules.misc_tools.ISA_tool import ISA 


# constants of physics
g0 = 9.80665            #[m/s^2]    O

#performance
v_cr = 300/3.6
v_stall = 40
roc_cr = 5
rod_cr = 3 # Rate of descend 
roc_hvr = 2
climb_gradient = 0.065
descent_slope = 0.04
h_cruise = 2400         #[m]        O
mission_dist = 400e3  # m
npax = 4                # Amount of passengers O
ax_target_climb = 0.5*g0   # PLACEHOLDER
ay_target_climb = 0.2*g0 # PLACEHOLDER
ax_target_descend = 0.5 * g0 # PLACEHOLDER
ay_target_descend = 0.2 * g0 # PLACEHOLDER

#atmospheric constants
atm = ISA(h_cruise)
rho_cr = atm.density()    #[kg/m^3]   O
p_cr = atm.pressure()           # pressure at cr O
t_cr = atm.temperature()      # Temperature at cr O
atm_stall = ISA(0)
t_stall = atm.temperature()
a_cr = atm.soundspeed()     #Speed of sound at cruise O
R = 287                 #[J/kg*K]   O
gamma = 1.4                  #        O

# Structures
beta_crash = 0.4 # crash diameter coefficient
E_mod = 70e9
poisson = 0.3
shear_mod =  26000000000.0
beta_wingbox = 1.42
sigma_yield = 430000000.0
ultimate_tensile_stress = 640000000.0
m_crip = 0.85
pb = 2.5
rho_material = 2710
g= 5
fuselage_margin = 0.2 
eigenfrequency_lim_pylon = 20
ARe = 2.75 # Aspect ration end of tail cone
n_tanks = 2 # The amount of tanks

s_p, s_y, e_0, e_d, v0, s0 = 0.5*10**6, 1.2*10**6, 0.038, 0.9, 9.1, 0.5





# Sea leavel atmospheric constants
rho_sl = atm.rho_SL            #[kg/m^3]   O
p_sl = atm.p_SL
T_sl = atm.T_SL
mhu_sl = atm.mu_SL
a_sl = atm.a_SL
mhu = atm.viscosity_dyn()           #[kg/m^3]   O   the dynamic viscosity


# Power
p_density = 7e3     # w totalEnergy/kg    ? # averaged from:  A review: high power density motors for electric vehicles

#standard masses
m_pl =  510  # kg

#airfoil
toc = 0.12 #NACA2412
xcm = 0.2973 # NACA2412
A_base = 0 #Assumed 0 base area
frac_lam_fus = 0.05
frac_lam_wing = 0.1
k = 0.634 * 10**(-5)  # Surface smoothness parameter

#airfoil V-tail
toc_tail = 0.12  # NACA 0012
xcm_tail = 0.2903

# Time constants for midterm
t_takeoff = 15.3
t_loiter = 20*60
t_landing = 15

#fuelcell input
VolumeDensityFuellCell = 3.25 #kW /l
PowerDensityFuellCell = 3.0 #kW/kg
effiencyFuellCell = 0.55

#Tank input
VolumeDensityTank = 0.5 #kg/l
EnergyDensityTank = 1.85 # kWh/kg

#battery input 
DOD = 0.8
dischargeEfficiency = 0.95
ChargingEfficiency = 0.7
EnergyDensityBattery = 0.3
PowerDensityBattery = 2
VolumeDensityBattery = 0.45


# Requirements
n_min_req = -1              # [-]       O   Min load factor
n_max_req = 2.5             # [-]       O   Max load factor
n_ult_req = 1.5*n_max_req       # [-]       O   Ultimate load factor
ub = 20.12              # [m/s]     0   Gust at Vb from EASA
uc = 15.24              # [m/s]     0   Gust at Vc from EASA
ud = 7.62               # [m/s]     0   Gust at Vd from EASA

# contingencies
oem_cont = 1.1

# Engine and properllors
diskloading = 120
n_engines = 6


#material properties
E_alu = 73e9
nu_alu = 0.33

E_composite = 100e9
sigma_compostie = 2000e6
rho_composite = 2000
