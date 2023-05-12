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
T0 = 288.15             #[K]        O
p0 = 101325             #[N/m^@]    O
R = 287                 #[J/kg*K]   O
n_ult = 2.5*1.5        # [-]       O   Ultimate load factor


class Propeller:
    eff_prop = 0.9          #[-]        ?
    eff_ductedfans = 0.8    #[-]        ?
    diskloading = 100       #[-]        ?

class Performance:
    V_cruise = 300/3.6     #[m/s]       O
    V_max = V_cruise*1.25   #[m/s]      O
    V_stall = 40            #[m/s]      ?
    loadfactor = 2          #[-]        ?
    h_hover = 30.5          #[m]        ~ EASE Requirements say that eVTOLs should hover to 30.5 meters before taking off
    ROC = 5                #[m/s]      ?
    G = 0.045               #[-]        ~ EASE Requirements say that eVTOLs should have a climb gradient of at least 4.5%


class Aero:
    CL_cruise = 0.5                #[-]        ?
    CLmax = 1.8            #[-]        ?
    CD = 0.07              #[-]        ?
    CD0 = 0.017
    e = 0.9                 #[-]        ?

class Wing:
    A = 6                   #[-]        ?
    StotS = 1.2             #[-]        ?

class Structure:
    payload = 510           #[kg]       0
    MTOMclassI = 2510       #[kg]       0
    OEMclassI = 2000        #[kg]       0