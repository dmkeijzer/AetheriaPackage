import numpy as np

# Physics
g = 9.80665                 # [m/s^2]
gamma = 1.4                 # [-]
R = 287                     # [J/kg/K]
rho_0 = 1.225               # [kg/m^3]

# Fuselage
w_fuselage = 1.38           # [m]
h_fuselage = 1.7            # [m]
l_nosecone = 2.5            # [m]
l_cylinder = 2              # [m]
l_tailcone = 2.7            # [m]
upsweep = 8.43*np.pi/180    # [Degrees]

# Aerodynamics
s1 = 0.5                    # Fraction of total wing area for the 1st wing [-]
s2 = 1-s1                   # Fraction of total wing area for the 2nd wing [-]
sweepc41 = 0                # Sweep angle at quarter chord for 1st wing [rad]
sweepc42 = 0                # Sweep angle at quarter chord for the 2nd wing
k = 0.634 * 10**(-5)        # Smooth paint from adsee 2 L2  smoothness factor[-]
flamf = 0.1                 # From ADSEE 2 L2 GA aircraft [-]
IF_f = 1                    # From ADSEE 2 L2 Interference factor fuselage [-]
IF_w = 1.1                  # From ADSEE 2 L2 Interference factor wing [-]
IF_v = 1.04                 # From ADSEE 2 L2 Interference factor vertical tail [-]
flamw = 0.35                # From ADSEE 2 L2 GA aircraft
Abase = 0                   # Base area of the fuselage [m2]
tc = 0.12                   # NACA0012 for winglets and Vtail [-]
xcm = 0.3                   # NACA0012 for winglets and Vtail [-]

# Propulsion
eff_prop = 0.83     # [-] Propeller efficiency during normal flight
eff_hover = 0.88    # [-] Propeller efficiency during hover
