import numpy as np
from math import *
from Airfoil_analysis_midterm2 import Cd
from Wing_design_midterm2 import winglet_dAR

# From BOX WING FUNDAMENTALS - A DESIGN PERSPECTIVE
# Oswald efficiency factor depending on the wing type
# h = 0.2, b = 1 can be used for a ratio of 0.2 which is a reasonable initial estimation
def e_factor(type,h, b, e_ref):
    """
    h = height difference between wings
    b = span
    """
    if type == 'box':
        ratio = h/b
        return e_ref * (0.44 + ratio* 2.219)/(0.44 + ratio * 0.9594)
    if type == 'tandem':
        ratio = h/b
        factor = 0.5 + (1-0.66 * ratio)/(2.1 + 7.4 * ratio)
        return e_ref * factor ** (-1)
    if type == 'normal':
        return e_ref

# Parasite Drag
"""
Cfe = 0.0045 for light twin wing aircraft ADSEE-I
Swet_ratio = 4 estimation using ADSEE-I
"""
def C_D_0(Swet_ratio, Cf): # ADSEE-I
    return Swet_ratio * Cf


# Parabolic Drag
def C_D(C_L, CL_CDmin, CD0, AR, e):
    return CD0 + (C_L-CL_CDmin )** 2 / (np.pi * AR * e)

# LD ratio from ADSEE-I
def LoD_ratio(phase, CD0, AR, e):
    if phase == 'cruise':
        return np.sqrt((np.pi * AR * e)/(4 * CD0))
    if phase == 'loiter':
        return np.sqrt((3 * np.pi * AR * e)/(16* CD0))

# C_L from ADSEE-I
def C_L(phase, CDmin, AR, e, C_LforCDmin):
    if phase == 'cruise':
        return np.sqrt(np.pi * AR * e * CDmin) +C_LforCDmin
    if phase == 'loiter':
        return np.sqrt(3 * np.pi * AR * e * CDmin) +C_LforCDmin

def e_OS(AR):
    return 1.78 * (1-0.045*AR**0.68)-0.64
# CD0 component build up

class componentdrag:
    def __init__(self, type, S_ref, l1, l2, l3, d, V_cr, rho, MAC, AR, e, M_cr, k, frac_lam_f, frac_lam_w, mu, tc, xcm,
                 sweepm, sweepLE, u, c_t,h, IF_f, IF_w,IF_v, C_L_minD, Abase, S_v, S_t, s1, s2, h_wl1, h_wl2):
        self.S_ref = S_ref
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.d = d
        self.V = V_cr
        self.rho = rho
        self.c = MAC
        self.AR = AR + s1*winglet_dAR(AR*2,h_wl1, np.sqrt(AR*S_ref))+ s2*winglet_dAR(AR*2,h_wl2, np.sqrt(AR*S_ref))
        self.e = e
        self.M = M_cr
        self.k = k
        self.frac_lamf = frac_lam_f
        self.frac_lamw = frac_lam_w
        self.mu = mu
        self.l = self.l1+self.l2+self.l3
        self.toc = tc
        self.xcm = xcm
        self.sweepm = sweepm
        self.u = u
        self.type = type
        self.IF_v = IF_v
        self.IF_w = IF_w
        self.IF_f = IF_f
        self.C_L_minD = C_L_minD
        self.Abase = Abase
        if self.type == 'box':
            self.S_c = c_t*h
        self.S_v = S_v
        self.S_t = S_t
        self.SweepLE = sweepLE
    def Swet_f(self):

        return (np.pi * self.d/4)* (((1/(3*self.l1**2))*((4*self.l1**2 +((self.d**2)/4))**1.5 -((self.d**3)/8)) ) -self.d + 4*self.l2 + 2 * np.sqrt(self.l3**2 + (self.d**2)/4 ))

    def Swet_v(self):

        if self.type =='box':
            return (self.S_ref+self.S_c + self.S_v) *2.14

        else:

            return (self.S_t+self.S_v)*2.14

    def Re_f(self):

        return min((self.rho * self.V * (self.l) / self.mu), 38.21 * (self.l / self.k) ** 1.053)

    def Re_w(self):

        return min((self.rho * self.V * (self.c) / self.mu), 38.21 * (self.c / self.k) ** 1.053)

    def Cf_f(self):

        Cflam = 1.328/np.sqrt(self.Re_f())
        Cfturb = 0.455/(((np.log10(self.Re_f()))**2.58)*(1 + 0.144 * self.M * self.M) ** 0.65 )

        return self.frac_lamf*Cflam + (1-self.frac_lamf)*Cfturb

    def Cf_w(self):
        Cflam = 1.328 / np.sqrt(self.Re_w())
        Cfturb = 0.455 / (((np.log10(self.Re_w())) ** 2.58) * (1 + 0.144 * self.M * self.M) ** 0.65)

        return self.frac_lamw * Cflam + (1-self.frac_lamw) * Cfturb

    def FF_f(self):
        f = self.l/self.d
        return 1+60/(f**3)+f/400
    def FF_w(self):
        return (1+0.6*self.toc/(self.xcm) + 100*self.toc**4) * (1.34*(self.M**0.18 ) *(np.cos(self.sweepm))**0.28)

    def CD0(self):

        self.CD0_f = (1/self.S_ref) * (self.Cf_f() *self.FF_f()*self.IF_f* self.Swet_f())
        self.CD0_v = (1 / self.S_ref) * (self.Cf_w() * self.FF_w() * self.IF_v * self.Swet_v())
        CD0 = (self.CD0_v + self.CD0_f)*1.05
        return CD0

    def CD_upsweep(self):

        return 3.83*(self.u**2.5)*np.pi*self.d**2/(4*self.S_ref)

    def CD_base(self):

        return (0.139 + 0.419*(self.M-0.161)**2) * self.Abase/(self.S_ref)

    def CDi(self, C_L):

        return ((C_L-self.C_L_minD)**2)/(np.pi *self.AR *self.e)

    def Cd_w(self, C_L):

        return self.IF_w * Cd(C_L/(np.cos(self.SweepLE)**2))

    def CD(self, C_L):

        return self.CD0()+self.CDi(C_L) + self.CD_base() + self.CD_upsweep() + self.Cd_w(C_L)

    def Drag(self, C_L):

        return self.CD(C_L) *0.5*self.rho* (self.V**2)*self.S_ref

    def Drag_polar(self):
        CDmin = self.CD0()+ self.CD_base() +self.CD_upsweep()
        K = 1/(np.pi*self.AR*self.e)
        return CDmin, K

    def CL_des(self):
        C_L_lst = np.arange(0,1.5,0.001)
        LD = C_L_lst/self.CD(C_L_lst)
        np.max(LD)
        index = np.where(LD==np.max(LD))
        return float(C_L_lst[index]), np.max(LD)