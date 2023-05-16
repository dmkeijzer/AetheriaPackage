import sys
import pathlib as pl
sys.path.append(str(list(pl.Path(__file__).parents)[2]))
import json
import os
import numpy as np
from input.GeneralConstants import *

os.chdir(str(list(pl.Path(__file__).parents)[2]))





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


# Define functions for all components of Class 2 drag estimations
def Reynolds(rho_cruise, V_cruise, mac, mu, k):
    """Returns the Reynold number

    :param rho_cruise: _description_
    :type rho_cruise: _type_
    :param V_cruise: _description_
    :type V_cruise: _type_
    :param mac: Mean aerodynamic chord
    :type mac: float
    :param mu: _description_
    :type mu: _type_
    :param k: surface factor in the order of 1e-5 and 1e-7
    :type k: float
    :return: Reyolds number
    :rtype: _type_
    """    
    return min((rho_cruise * V_cruise * mac / mu), 38.21 * (mac / k) ** 1.053)

def Mach_cruise(V_cruise, gamma, R, T_cruise):
    """_summary_

    :param V_cruise: _description_
    :type V_cruise: _type_
    :param gamma: _description_
    :type gamma: _type_
    :param R: _description_
    :type R: _type_
    :param T_cruise: _description_
    :type T_cruise: _type_
    :return: _description_
    :rtype: _type_
    """    
    a = np.sqrt(gamma * R * T_cruise)
    return V_cruise / a

def FF_fus(l, d):
    """_summary_

    :param l: _description_
    :type l: _type_
    :param d: _description_
    :type d: _type_
    :return: _description_
    :rtype: _type_
    """    
    f = l / d
    return 1 + 60 / (f ** 3) + f / 400

def FF_wing(toc, xcm, M, sweep_m):
    """_summary_

    :param toc: _description_
    :type toc: _type_
    :param xcm: _description_
    :type xcm: _type_
    :param M: _description_
    :type M: _type_
    :param sweep_m: _description_
    :type sweep_m: _type_
    :return: _description_
    :rtype: _type_
    """    
    return (1 + 0.6 * toc / xcm + 100 * toc * 4) * (1.34 * (M * 0.18) * (np.cos(sweep_m)) * 0.28)

def S_wet(d, l1, l2, l3):
    """_summary_

    :param d: _description_
    :type d: _type_
    :param l1: _description_
    :type l1: _type_
    :param l2: _description_
    :type l2: _type_
    :param l3: _description_
    :type l3: _type_
    :return: _description_
    :rtype: _type_
    """    
    return (np.pi * d / 4) * (((1 / (3 * l1 ** 2)) * ((4 * l1 ** 2 + ((d ** 2) / 4)) ** 1.5 - ((d ** 3) / 8))) - d + 4 * l2 + 2 * np.sqrt(l3 ** 2 + (d ** 2) / 4))

def CD_upsweep(u, d, S):
    """_summary_

    :param u: _description_
    :type u: _type_
    :param d: _description_
    :type d: _type_
    :param S: _description_
    :type S: _type_
    :return: _description_
    :rtype: _type_
    """    
    return 3.83 * (u * 2.5) * np.pi * d * 2 / (4 * S)

def CD_base(M, A_base, S):
    """_summary_

    :param M: _description_
    :type M: _type_
    :param A_base: _description_
    :type A_base: _type_
    :param S: _description_
    :type S: _type_
    :return: _description_
    :rtype: _type_
    """    
    return (0.139 + 0.419 * (M - 0.161) ** 2) * A_base / S

def C_fe_fus(frac_lam_fus, Reynolds):
    """_summary_

    :param frac_lam_fus: _description_
    :type frac_lam_fus: _type_
    :param Reynolds: _description_
    :type Reynolds: _type_
    :return: _description_
    :rtype: _type_
    """    
    C_f_lam = 1.328 / np.sqrt(Reynolds)
    C_f_turb = 0.455 / (((np.log10(Reynolds)) ** 2.58)
                        * (1 + 0.144 * M ** 2) ** 0.65)
    return frac_lam_fus * C_f_lam + (1 - frac_lam_fus) * C_f_turb

def C_fe_wing(frac_lam_wing, Reynolds):
    """_summary_

    :param frac_lam_wing: _description_
    :type frac_lam_wing: _type_
    :param Reynolds: _description_
    :type Reynolds: _type_
    :return: _description_
    :rtype: _type_
    """    
    C_f_lam = 1.328 / np.sqrt(Reynolds)
    C_f_turb = 0.455 / (((np.log10(Reynolds)) ** 2.58)
                        * (1 + 0.144 * M ** 2) ** 0.65)
    return frac_lam_wing * C_f_lam + (1 - frac_lam_wing) * C_f_turb


def CD_fus(C_fe_fus, FF_fus, S_wet_fus):
    """_summary_

    :param C_fe_fus: _description_
    :type C_fe_fus: _type_
    :param FF_fus: _description_
    :type FF_fus: _type_
    :param S_wet_fus: _description_
    :type S_wet_fus: _type_
    :return: _description_
    :rtype: _type_
    """    
    IF_fus = 1.0        # From WIGEON script
    return C_fe_fus * FF_fus * IF_fus * S_wet_fus

def CD_wing(C_fe_wing, FF_wing, S_wet_wing):
    """_summary_

    :param C_fe_wing: _description_
    :type C_fe_wing: _type_
    :param FF_wing: _description_
    :type FF_wing: _type_
    :param S_wet_wing: _description_
    :type S_wet_wing: _type_
    :return: _description_
    :rtype: _type_
    """    
    IF_wing = 1.1       # From WIGEON script
    return C_fe_wing * FF_wing * IF_wing * S_wet_wing


def CD0(S, CD_fus, CD_wing, CD_upsweep, CD_base):
    """_summary_

    :param S: _description_
    :type S: _type_
    :param CD_fus: _description_
    :type CD_fus: _type_
    :param CD_wing: _description_
    :type CD_wing: _type_
    :param CD_upsweep: _description_
    :type CD_upsweep: _type_
    :param CD_base: _description_
    :type CD_base: _type_
    :return: _description_
    :rtype: _type_
    """    
    return (1 / S) * (CD_fus + CD_wing) + CD_upsweep + CD_base


def CDi(CL, A, e):
    """_summary_

    :param CL: _description_
    :type CL: _type_
    :param A: _description_
    :type A: _type_
    :param e: _description_
    :type e: _type_
    :return: _description_
    :rtype: _type_
    """    
    return CL**2 / (np.pi * A * e)

def CD(CD0, CDi):
    """_summary_

    :param CD0: _description_
    :type CD0: _type_
    :param CDi: _description_
    :type CDi: _type_
    :return: _description_
    :rtype: _type_
    """    
    return CD0 + CDi

def lift_over_drag(CL_output, CD_output):
    """_summary_

    :param CL_output: _description_
    :type CL_output: _type_
    :param CD_output: _description_
    :type CD_output: _type_
    :return: _description_
    :rtype: _type_
    """    
    return CL_output / CD_output


def Oswald_eff(A):
    """_summary_

    :param A: _description_
    :type A: _type_
    :return: _description_
    :rtype: _type_
    """    
    return 1.78 * (1 - 0.045 * A**0.68) - 0.64

def Oswald_eff_tandem(b1, b2, h, b, Oswald_eff):
    """_summary_

    :param b1: _description_
    :type b1: _type_
    :param b2: _description_
    :type b2: _type_
    :param h: _description_
    :type h: _type_
    :param b: _description_
    :type b: _type_
    :param Oswald_eff: _description_
    :type Oswald_eff: _type_
    :return: _description_
    :rtype: _type_
    """    
    b_avg = (b1 + b2) / 2
    factor = 0.5 + (1 - 0.66 * (h / b_avg)) / (2.1 + 7.4 * (h / b_avg))
    return factor * Oswald_eff

