import Aero_tools as at
import numpy as np
from constants import *


class BL:

    def __init__(self, Sw, b, V, xc_prop, c_r, taper, mu, density, D_prop, laminar=True):
        """
        :param Sw: Wing surface area [m^2]
        :param b: Wing span (FULL wing) [m]
        :param V: Freestream velocity [m/s]
        :param xc_prop: % of the chord at which the propulsion system is located [-]
        :param c_r: Root chord [m]
        :param taper: Taper ratio [-]
        :param mu: Dynamic viscosity [N*s/m^2]
        :param density: Density [kg/m^3]
        :param D_prop: Diameter of one propeller [m]
        :param laminar: Boolean, laminar flow true or false
        """
        self.laminar = laminar
        self.nu = mu/density
        self.density = density
        self.mu = mu
        self.xc_prop = xc_prop
        self.b = b
        self.S = Sw
        self.taper = taper
        self.c_r = c_r
        self.D_prop = D_prop
        self.V = V

    def c(self, y):
        return self.c_r * (1 - (1-self.taper)/self.b * 2*y)

    def L(self, y):
        return self.c(y) * self.xc_prop

    def Re(self, y):
        return self.density * self.V * self.L(y) / self.mu

    def BL_height(self, y):
        if self.laminar:
            return 5.2 * np.sqrt(self.L(y) / self.Re(y))  # https://ocw.tudelft.nl/wp-content/uploads/AE1101-Aero-4.pdf
        else:
            return 0.37 * self.L(y) / self.Re(y)**(1/5)  # https://ocw.tudelft.nl/wp-content/uploads/AE1101-Aero-4.pdf

    def ratio_BL_D(self, y):
        return self.BL_height(y)/self.D_prop


class LE_prop:

    def __init__(self, ve, v0, L, Sw):
        """
        :param ve: Fan exit speed [m/s]
        :param v0:
        :param L: Lift in the wing without propulsion effects [N]
        :param Sw: Surface area of the wing [m^2]
        # :param WS: Wing loading [N/m^2]
        """
        self.ve = ve
        self.v0 = v0
        self.L = L
        self.S = Sw
        # self.WS = WS

    def S1(self):
        return self.S * (self.v0/self.ve)**2

    def S_ratio(self):
        return self.S1()/self.S


# class Noise:
#
#     def __init__(self, A, n_prop, ve, T):
#         """
#         :param A: Total disk area of the propellers [m^2]
#         :param n_prop: Number of propulsors [-]
#         :param ve: Jet velocity of propeller [m/s]
#         :param T: Total thrust [N]
#         """
#         self.A = A
#         self.n = n_prop
#         self.ve = ve
#         self.T = T
#
#     def Delta_noise_jet(self):
#         return 10*np.log(2)
