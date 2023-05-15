from input.GeneralConstants import *
import numpy as np
import json
# import CL_cruise from json files

# Define the directory and filenames of the JSON files
dict_directory = "path_to_directory"
dict_names = ["dict1.json", "dict2.json", "dict3.json"]


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
        self.rho_SL = 1.225                                 # [kg/m^3]  Sea level density
        self.p_SL   = 101325                                # [Pa]      Sea level pressure
        self.T_SL   = 288.15 + T_offset                     # [K]       Sea level temperature
        self.mu_SL  = 1.7894E-5                             # [kg/m/s] Sea Level Dynamic Viscosity 1.81206E-5
        self.a_SL   = np.sqrt(self.gamma*self.R*self.T_SL)  # [m/s] Sea level speed of sound

        self.h = h  # [m]       Altitude

        # Throw an error if the specified altitude is outside of the troposphere
        if np.any(h) > 11000:
            raise ValueError('The specified altitude is outside the range of this class')

        self.T = self.T_SL + self.a * self.h  # [K] Temperature at altitude h, done here because it is used everywhere

    def temperature(self):
        return self.T

    def pressure(self):
        p = self.p_SL * (self.T / self.T_SL) ** (-self.g0 / (self.a * self.R))
        return p

    def density(self):
        rho = self.rho_SL * (self.T / self.T_SL) ** (-self.g0 / (self.a * self.R) - 1)
        return rho

    def soundspeed(self):
        a = self.a_SL * np.sqrt(self.T/self.T_SL)
        return a

    def viscosity_dyn(self):
        mu = self.mu_SL * (self.T / self.T_SL) ** (1.5) * (self.T_SL + 110.4) / (self.T + 110.4)
        # 1.458E-6 * self.T ** 1.5 / (self.T + 110.4) # Sutherland Law, using Sutherland's constant S_mu = 110.4 for air
        return mu



# Loop through the JSON files
for dict_name in dict_names:
    # Load data from JSON file
    with open(dict_directory + "\\" + dict_name, "r") as jsonFile:
        data = json.load(jsonFile)

    # Define functions for all components of Class 2 drag estimations

    def CD0():
        return (1 / S) * (CD_fus() + CD_wing()) + CD_upsweep() + CD_base()


    def CD_fus():
        IF_fus = 1.0        # From WIGEON script
        return C_fe_fus() * FF_fus() * IF_fus * S_wet_fus

    def CD_wing():
        IF_wing = 1.1       # From WIGEON script
        return C_fe_wing() * FF_wing() * IF_wing * S_wet_wing

    def FF_fus():
        f = l / d
        return 1 + 60 / (f ** 3) + f / 400

    def FF_wing():
        return (1 + 0.6 * toc / (xcm) + 100 * toc * 4) * (1.34 * (M * 0.18) * (np.cos(sweep_m)) * 0.28)

    def C_fe_fus():
        frac_lam_fus = 0.05              # From ADSEE II course
        C_f_lam = 1.328 / np.sqrt(Reynolds())
        C_f_turb = 0.455 / (((np.log10(Reynolds())) ** 2.58) * (1 + 0.144 * M ** 2) ** 0.65)
        return frac_lam_fus * C_f_lam + (1 - frac_lam_fus) * C_f_turb

    def C_fe_wing():
        frac_lam_wing = 0.10            # From ADSEE II course
        C_f_lam = 1.328 / np.sqrt(Reynolds())
        C_f_turb = 0.455 / (((np.log10(Reynolds())) ** 2.58) * (1 + 0.144 * M ** 2) ** 0.65)
        return frac_lam_wing * C_f_lam + (1 - frac_lam_wing) * C_f_turb

    def Reynolds():
        return min((rho_cruise * V_cruise * (l) / mu), 38.21 * (l / k) ** 1.053)

    def S_wet():
        return (np.pi * d / 4) * (((1 / (3 * l1 ** 2))((4 * l1 ** 2 + ((d ** 2) / 4))**1.5 - ((d**3) / 8))) - d + 4 * l2 + 2 * np.sqrt(l3**2 + (d ** 2) / 4))

    def CD_upsweep():
        return 3.83 * (u * 2.5) * np.pi * d * 2 / (4 * S)

    def CD_base():
        return (0.139 + 0.419 * (M - 0.161) ** 2) * A_base / (S)

    def CDi():
        return CL**2 / (np.pi * A * e)

    def CD():
        return CD0() + CDi()

    def lift_over_drag(CL):
        return CL / CD()

    def Mach_cruise():
        a = np.sqrt(gamma * R * T_cruise)
        return V_cruise / a

    def Oswald_eff():
        return 1.78 * (1 - 0.045 * A**0.68) - 0.64

    def Oswald_eff_tandem():
        b = (b1 + b2) / 2
        factor = 0.5 + (1 - 0.66 * (h/(b)) / (2.1 + 7.4 * (h/b))
        return factor * Oswald_eff()
