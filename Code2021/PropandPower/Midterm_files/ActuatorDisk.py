from constants import *
import numpy as np
import Aero_tools as at

ISA = at.ISA(h_cruise)

class ActDisk:
    def __init__(self, TW_ratio, MTOM, v_e_h, v_cruise, D, DiskLoad):
        """
        :param TW_ratio: Thrust to weight ratio [-]
        :param MTOM: Maximum take-off MASS [kg]
        :param v_e_h: Jet speed at hover [m/s] (obtained from disk loading graph)
        :param v_cruise: Cruise speed [m/s]
        :param D: Total drag at cruise [N]
        :param DiskLoad: Disk loading [kg/m^2]
        """
        self.TW_ratio = TW_ratio
        self.MTOM = MTOM
        self.v_e_h = v_e_h
        self.v_cr = v_cruise
        self.D = D
        self.T_cruise = D
        self.g0 = 9.80665
        self.rho0 = 1.225
        self.A = MTOM/DiskLoad
        self.T_hover = MTOM*self.g0

    # def A_disk(self):
    #     return 2*self.T_hover / (self.rho0 * (self.v_e_h**2 - 0))

    # def v_0_hover(self):
    #     return np.sqrt(2*self.T_hover/(self.rho0*self.A) - self.v_e_h**2)

    def v_e_hover(self):
        return np.sqrt(2*self.T_hover/(self.rho0*self.A) + 0)

    def v_e_cr(self):
        # return np.sqrt(2*self.T_cruise/(ISA.density() * self.A_disk()) + self.v_cr**2)
        return np.sqrt(2 * self.T_cruise / (ISA.density() * self.A) + self.v_cr ** 2)

    def eff_cruise(self):
        return 2/(1 + self.v_e_cr()/self.v_cr)

    # def eff_hover(self):
    #     return 2

    def P_ideal(self):
        # return 0.5*self.T_cruise*self.v_cr * ((self.T_cruise/(self.A_disk() * self.v_cr**2 * (ISA.density()/2)) + 1)**2 + 1)
        return 0.5 * self.T_cruise * self.v_cr * ((self.T_cruise / (self.A * self.v_cr ** 2 * (ISA.density() / 2)) + 1) ** 2 + 1)

    def P_actual(self):
        return self.P_ideal()*1.15
