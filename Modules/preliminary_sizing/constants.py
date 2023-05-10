'''
---------- EXPLANATION ----------
In these constants the unit is included and the definitiveness
(if that's a word) of the variable.
So an O means that this value can be taken as true, 
so for example the atmospheric constants won't change 
and the cruise speed has already been defined by requirements. 
An ~ means it's defined but maybe not needed and a ? means it's a guess.


'''
g0 = 9.80665            #[m/s^2]    O
rho0 = 1.225            #[kg/m^3]   O
rho300 = 1.19011        #[kg/m^3]   O
T0 = 288.15             #[K]        O
p0 = 101325             #[N/m^@]    O
R = 287                 #[J/kg*K]   O


eff_prop = 0.8          #[-]        ?
ROC = 10                #[m/s]      ?

CL = 1.5                #[-]        ?
CD = 0.3                #[-]        ?
A = 6                   #[-]        ?
e = 0.9                 #[-]        ?

V_cruise = 300/3.6      #[m/s]      O
V_max = V_cruise*1.25   #[m/s]      O
V_stall = 40            #[m/s]      ?

loadfactor = 2          #[-]        ?

h_hover = 30.5          #[m]        ~ EASE Requirements say that eVTOLs should hover to 30.5 meters before taking off
no_engines = 6          #[m]        ?
