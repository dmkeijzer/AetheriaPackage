import sys
import pathlib as pl
sys.path.append(str(list(pl.Path(__file__).parents)[2]))
import json
import os
import numpy as np
from input.GeneralConstants import *

os.chdir(str(list(pl.Path(__file__).parents)[2]))



# import CL_cruise from json files

# Define the directory and filenames of the JSON files
dict_directory = "input"
dict_names = ["J1_constants.json", "L1_constants.json", "W1_constants.json"]


###################      Move class ISA to different location, ISA is needed for T_cruise       #################

class ISA:
    """
    Calculates the atmospheric parameters at a specified altitude h (in meters).
    An offset in sea level temperature can be specified to allow for variations.

    Note: Since our aircraft probably doesn't fly too high, this is only valid in the troposphere

    Verified by comparison to: https://www.digitaldutch.com/atmoscalc/
    """

    def __init__(self, h, T_offset=0):

        # Constants
        self.a = -0.0065    # [K/m]     Temperature lapse rate
        self.g0 = 9.80665   # [m/s^2]   Gravitational acceleration
        self.R = 287        # [J/kg K]  Specific gas constant
        self.gamma = 1.4    # [-]       Heat capacity ratio

        # Sea level values
        # [kg/m^3]  Sea level density
        self.rho_SL = 1.225
        # [Pa]      Sea level pressure
        self.p_SL = 101325
        # [K]       Sea level temperature
        self.T_SL = 288.15 + T_offset
        # [kg/m/s] Sea Level Dynamic Viscosity 1.81206E-5
        self.mu_SL = 1.7894E-5
        # [m/s] Sea level speed of sound
        self.a_SL = np.sqrt(self.gamma*self.R*self.T_SL)

        self.h = h  # [m]       Altitude

        # Throw an error if the specified altitude is outside of the troposphere
        if np.any(h) > 11000:
            raise ValueError(
                'The specified altitude is outside the range of this class')

        # [K] Temperature at altitude h, done here because it is used everywhere
        self.T = self.T_SL + self.a * self.h

    def temperature(self):
        return self.T

    def pressure(self):
        p = self.p_SL * (self.T / self.T_SL) ** (-self.g0 / (self.a * self.R))
        return p

    def density(self):
        rho = self.rho_SL * \
            (self.T / self.T_SL) ** (-self.g0 / (self.a * self.R) - 1)
        return rho

    def soundspeed(self):
        a = self.a_SL * np.sqrt(self.T/self.T_SL)
        return a

    def viscosity_dyn(self):
        mu = self.mu_SL * (self.T / self.T_SL) ** (1.5) * \
            (self.T_SL + 110.4) / (self.T + 110.4)
        # 1.458E-6 * self.T ** 1.5 / (self.T + 110.4) # Sutherland Law, using Sutherland's constant S_mu = 110.4 for air
        return mu


# Loop through the JSON files
for dict_name in dict_names:
    # Load data from JSON file
    with open(os.path.join(dict_directory, dict_name)) as jsonFile:
        data = json.load(jsonFile)
    CL=0.3
    # Define functions for all components of Class 2 drag estimations
    def Reynolds(rho_cruise, V_cruise, l, mu, k):
        return min((rho_cruise * V_cruise * l / mu), 38.21 * (l / k) ** 1.053)
    
    def Mach_cruise(V_cruise, gamma, R, T_cruise):
        a = np.sqrt(gamma * R * T_cruise)
        return V_cruise / a

    def FF_fus(l, d):
        f = l / d
        return 1 + 60 / (f ** 3) + f / 400

    def FF_wing(toc, xcm, M, sweep_m):
        return (1 + 0.6 * toc / xcm + 100 * toc * 4) * (1.34 * (M * 0.18) * (np.cos(sweep_m)) * 0.28)

    def S_wet(d, l1, l2, l3):
        return (np.pi * d / 4) * (((1 / (3 * l1 ** 2)) * ((4 * l1 ** 2 + ((d ** 2) / 4)) ** 1.5 - ((d ** 3) / 8))) - d + 4 * l2 + 2 * np.sqrt(l3 ** 2 + (d ** 2) / 4))

    def CD_upsweep(u, d, S):
        return 3.83 * (u * 2.5) * np.pi * d * 2 / (4 * S)

    def CD_base(M, A_base, S):
        return (0.139 + 0.419 * (M - 0.161) ** 2) * A_base / S

    def C_fe_fus(frac_lam_fus, Reynolds):
        C_f_lam = 1.328 / np.sqrt(Reynolds)
        C_f_turb = 0.455 / (((np.log10(Reynolds)) ** 2.58)
                            * (1 + 0.144 * M ** 2) ** 0.65)
        return frac_lam_fus * C_f_lam + (1 - frac_lam_fus) * C_f_turb

    def C_fe_wing(frac_lam_wing, Reynolds):
        C_f_lam = 1.328 / np.sqrt(Reynolds)
        C_f_turb = 0.455 / (((np.log10(Reynolds)) ** 2.58)
                            * (1 + 0.144 * M ** 2) ** 0.65)
        return frac_lam_wing * C_f_lam + (1 - frac_lam_wing) * C_f_turb

    
    def CD_fus(C_fe_fus, FF_fus, S_wet_fus):
        IF_fus = 1.0        # From WIGEON script
        return C_fe_fus * FF_fus * IF_fus * S_wet_fus

    def CD_wing(C_fe_wing, FF_wing, S_wet_wing):
        IF_wing = 1.1       # From WIGEON script
        return C_fe_wing * FF_wing * IF_wing * S_wet_wing

    
    def CD0(S, CD_fus, CD_wing, CD_upsweep, CD_base):
        return (1 / S) * (CD_fus + CD_wing) + CD_upsweep + CD_base

    
    def CDi(CL, A, e):
        return CL**2 / (np.pi * A * e)

    def CD(CD0, CDi):
        return CD0 + CDi

    def lift_over_drag(CL_output, CD_output):
        return CL_output / CD_output


    def Oswald_eff(A):
        return 1.78 * (1 - 0.045 * A**0.68) - 0.64

    def Oswald_eff_tandem(b1, b2, h, b, Oswald_eff):
        b_avg = (b1 + b2) / 2
        factor = 0.5 + (1 - 0.66 * (h / b_avg)) / (2.1 + 7.4 * (h / b_avg))
        return factor * Oswald_eff

