import numpy as np
import Aero_tools as at


class ActDisk:
    def __init__(self, TW_ratio, MTOM, v_e_h, v_cruise, D, DiskLoad, rho, rho0, g0):
        """
        :param TW_ratio: Thrust to weight ratio [-]
        :param MTOM: Maximum take-off MASS [kg]
        :param v_e_h: Jet speed at hover [m/s] (obtained from disk loading graph)
        :param v_cruise: Cruise speed [m/s]
        :param D: Total drag at cruise [N]
        :param DiskLoad: Disk loading [kg/m^2]
        :param rho: Density [kg/m^3]
        """
        self.TW_ratio = TW_ratio
        self.MTOM = MTOM
        self.v_e_h = v_e_h
        self.v_cr = v_cruise
        self.D = D
        self.T_cruise = D
        self.rho = rho
        self.g0 = g0
        self.rho0 = rho0
        self.A = MTOM/DiskLoad
        self.T_hover = MTOM*self.g0

    def v_e_hover(self):
        return np.sqrt(2*self.T_hover/(self.rho0*self.A) + 0)

    def v_e_cr(self):
        return np.sqrt(2 * self.T_cruise / (self.rho * self.A) + self.v_cr**2)

    def eff_cruise(self):
        return 2/(1 + self.v_e_cr()/self.v_cr)

    def P_ideal(self):
        return 0.5 * self.T_cruise * self.v_cr * ((self.T_cruise / (self.A * self.v_cr ** 2 * (self.rho / 2)) + 1) ** 2 + 1)

    def P_actual(self):
        return self.P_ideal()*1.15

class ActDisk_verif:
    def __init__(self, v_cruise, T_cr_tot, rho, A_tot):
        """
        :param TW_ratio: Thrust to weight ratio [-]
        :param MTOM: Maximum take-off MASS [kg]
        :param v_e_h: Jet speed at hover [m/s] (obtained from disk loading graph)
        :param v_cruise: Cruise speed [m/s]
        :param T_cr: Total thrust needed at cruise [N]
        :param DiskLoad: Disk loading [kg/m^2]
        :param rho: Density [kg/m^3]
        :param A_tot: Total disk area [m^2]
        :param N_prop: Number of propellers [-]
        """
        # self.TW_ratio = TW_ratio
        # self.MTOM = MTOM
        # self.v_e_h = v_e_h
        self.v_cr = v_cruise
        # self.D = D
        self.T_cruise = T_cr_tot
        self.rho = rho
        self.A_tot = A_tot
        # self.g0 = g0
        # self.rho0 = rho0
        # self.A = MTOM/DiskLoad
        # self.T_hover = MTOM*self.g0

    # def v_e_hover(self):
    #     return np.sqrt(2*self.T_hover/(self.rho0*self.A) + 0)

    def v_e_cr(self):
        return np.sqrt(2 * self.T_cruise / (self.rho * self.A_tot) + self.v_cr**2)

    # def eff_cruise(self):
    #     return 2/(1 + self.v_e_cr()/self.v_cr)
    #
    # def P_ideal(self):
    #     return 0.5 * self.T_cruise * self.v_cr * ((self.T_cruise / (self.A * self.v_cr ** 2 * (self.rho / 2)) + 1) ** 2 + 1)
    #
    # def P_actual(self):
    #     return self.P_ideal()*1.15
