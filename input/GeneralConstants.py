'''
---------- EXPLANATION ----------
In these constants the unit is included and the definitiveness
(if that's a word) of the variable.
So an O means that this value can be taken as true, 
so for example the atmospheric constants won't change 
and the cruise speed has already been defined by requirements. 
An ~ means it's defined but maybe not needed and a ? means it's a guess.


'''

h_cruise = 400          #[m]        O
g0 = 9.80665            #[m/s^2]    O
rho0 = 1.225            #[kg/m^3]   O
rho_cruise = 1.19011    #[kg/m^3]   O
mhu = 1.19011           #[kg/m^3]   O   the dynamic viscosity
T0 = 288.15             #[K]        O
p0 = 101325             #[N/m^@]    O
R = 287                 #[J/kg*K]   O
npax = 4                # Amount of passengers 0
gamma = 1.4             #


# Power
p_density = 3.117e3     # w totalEnergy/kg    ? # averaged from:  A review: high power density motors for electric vehicles
DOD = 0.8
ChargingEfficiency = 0.7

#performance
v_cr = 300/3.6
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
