from dataclasses import dataclass
import numpy as np
@dataclass
class Fuselage():
    cockpit_parabol_length: float 
    cylindrical_fuselage_length: float = None
    conical_tail_length: float = None
    length: float = cockpit_parabol_length + cylindrical_fuselage_length + conical_tail_length
    diameter: float = None
    fraction_laminar_flow: float = None
    surface_friction_coefficient:float = None
    interference_factor: float = None
    upsweep_angle: float = None
    base_area: float = None

@dataclass
class Wing():
    surface: float
    taper: float
    aspectratio: float
    span: float = None
    chord_root: float = None
    chord_tip: float = None
    chord_mac: float = None
    y_mac: float = None
    sweep_LE: float = None ### check that this is implement corrcetly in the function
    quarterchord_sweep: float = None
    mac_sweep: float = None
    X_lemac: float = None
    effective_aspectratio: float = None
    effective_span: float = None
    oswald_factor: float = None
    fraction_laminar_flow: float = None
    interference_factor: float = None

@dataclass
class FlightParam():
    mach_cruise: float
    rho_cruise: float
    viscosity: float
    V_cruise: float
    V_hover: float
    V_stall: float
    V_max: float
    Reynolds_no: float

@dataclass
class Airfoil:
    C_l: float
    C_L: float
    C_d0: float
    C_D0: float
    C_Lmax: float
    C_L_alpha: float
    C_L_cruise: float
    thickness_to_chord: float
    position_max_t_relative: float 

@dataclass
class Winglet():
    height: float
    smoothness: float


class componentdrag:
    def __init__(self, type, fuselage: Fuselage, wing1: Wing, wing2: Wing, airfoil: Airfoil, flightparam: FlightParam, winglet1: Winglet, winglet2: Winglet, S_ref, MAC,s1,s2,S_v,C_L_minD, height_between_wings, ):
        self.S_ref = S_ref
        self.l1 = fuselage.cockpit_parabol_length #parabolic cockpit section
        self.l2 = fuselage.cylindrical_fuselage_length   #cylindrical fuselage length 
        self.l3 = fuselage.conical_tail_length # conical tail length 
        self.d = fuselage.diameter #fuselage diameter
        self.V = flightparam.V_cruise
        self.rho = flightparam.rho_cruise
        self.c = MAC #mac
        self.b = np.sqrt((0.5*(s1*wing1.aspectratio +s2*wing2.aspectratio))*S_ref)
        self.h_wl1 = winglet1.height #winglet height 1
        self.h_wl2 = winglet2.height #winglet height 2
        self.AR = 0.5*(s1*(wing1.aspectratio*winglet_factor(winglet1.height , np.sqrt(wing1.aspectratio*S_ref*s1), winglet1.smoothness))+ s2*(wing2.aspectratio*winglet_factor(winglet2.height, np.sqrt(wing2.aspectratio*S_ref*s2), winglet2.smoothness)))
        #0.5*(s1*(AR1+winglet_dAR(AR1,self.h_wl1, np.sqrt(AR1*S_ref*s1)))+ s2*(AR2+winglet_dAR(AR2,self.h_wl2, np.sqrt(AR2*S_ref*s2))))
        self.e =  #oswald efficiency factor
        self.M = flightparam.mach_cruise #cruise mac number
        self.k = fuselage.sur # surfaces friction value of zoiets
        self.frac_lamf = fuselage.fraction_laminar_flow #fraction laminer flow fuselage
        self.frac_lamw = wing1.fraction_laminar_flow#fraction laminer flow wing
        self.mu = flightparam.viscosity #viscosity
        self.l = fuselage.length #fuselage length 
        self.toc = airfoil.thickness_to_chord #t/c
        self.xcm = airfoil.position_max_t_relative #position maximum thickness wing
        self.sweepm = sweepm 
        self.u = fuselage.upsweep_angle #upsweep angle [rad]
        self.type = type
        self.IF_v = IF_v #interference factor vertical tail
        self.IF_w = wing1.interference_factor #interference factor wing
        self.IF_f = fuselage.interference_factor #interference factor 
        self.Abase = fuselage.base_area #fuselage base area
 
        self.S_v = S_v  #surface area vertical tail
        self.S_t = 0.5*(1.4*wing1.chord_tip )*winglet1.height*2 + 0.5*(1.4*wing2.chord_tip) *winglet2.height
        self.SweepLE = wing1 #sweep LE
        self.C_L_minD = C_L_minD / (np.cos(self.SweepLE) ** 2) #cl at minimum cd
        self.h = height_between_wings #height between the wings
        self.k_wl = winglet1.smoothness #smoothiness factor winglet

    def e_OS(aspectratio):
        AR = [4, 6, 8, 10]
        e = [0.997, 0.993, 0.990, 0.98]
        curve = np.interp1d(AR, e)
        if aspectratio < 4:
            return 0.997
        else:
            return curve(aspectratio)
    
    def e_factor(self, type, h, b):
        ratio = h / b
        if type == 'box':
            return self.e_OS() * (0.44 + ratio * 2.219) / (0.44 + ratio * 0.9594)
        if type == 'tandem':
            factor = 0.5 + (1 - 0.66 * ratio) / (2.1 + 7.4 * ratio)
            return self.e_OS() * factor ** (-1)
        if type == 'normal':
            return (1 + (1 - 0.66 * ratio) / (1.05 + 3.7 * ratio)) ** (-1)
    
    def Swet_f(fuselage: Fuselage):
        return (np.pi * fuselage.diameter / 4) * (((1 / (3 * fuselage.cockpit_parabol_length ** 2)) * ((4 * fuselage.cockpit_parabol_length ** 2 + ((fuselage.diameter ** 2) / 4)) ** 1.5 - ((fuselage.diameter ** 3) / 8))) - fuselage.diameter + 4 * fuselage.cylindrical_fuselage_length + 2 * np.sqrt( fuselage.conical_tail_length ** 2 + (fuselage.diameter ** 2) / 4))
    
    def Swet_v(self, type, S_ref, S_c, S_v, S_t):
        if type == 'box':
            return (S_ref + S_c + S_v) * 2.14
        else:
            return (S_t + S_v) * 2.14
    
    def Re_f(self, rho, V, l, mu, k):
        return min((rho * V * l / mu), 38.21 * (l / k) ** 1.053)
    
    def Re_w(self, rho, V, c, mu, k):
        return min((rho * V * c / mu), 38.21 * (c / k) ** 1.053)
    
    def Cf_f(self, frac_lamf, Re_f, M):
        Cflam = 1.328 / np.sqrt(Re_f)
        Cfturb = 0.455 / (((np.log10(Re_f)) ** 2.58) * (1 + 0.144 * M * M) ** 0.65)
        return frac_lamf * Cflam + (1 - frac_lamf) * Cfturb
    
    def Cf_w(self, frac_lamw, Re_w, M):
        Cflam = 1.328 / np.sqrt(Re_w)
        Cfturb = 0.455 / (((np.log10(Re_w)) ** 2.58) * (1 + 0.144 * M * M) ** 0.65)
        return frac_lamw * Cflam + (1 - frac_lamw) * Cfturb
    
    def FF_f(self, l, d):
        f = l / d
        return 1 + 60 / (f ** 3) + f / 400
    
    def FF_w(self, toc, xcm, M, sweepm):
        return (1 + 0.6 * toc / xcm + 100 * toc ** 4) * (1.34 * (M ** 0.18) * (np.cos(sweepm)) ** 0.28)
    
    def CD0(self, S_ref, Cf_f, FF_f, IF_f, Swet_f, S_c, Cf_w, FF_w, IF_v, Swet_v):
        CD0_f = (1 / S_ref) * (Cf_f * FF_f * IF_f * Swet_f)
        CD0_v = (1 / S_ref) * (Cf_w * FF_w * IF_v * Swet_v)
        CD0 = (CD0_v + CD0_f) * 1.05
        return CD0
    
    def CD_upsweep(self, u, d, S_ref):
        return 3.83 * (u ** 2.5) * np.pi * (d ** 2) / (4 * S_ref)
    
    def CD_base(self, M, Abase, S_ref):
        return (0.139 + 0.419 * (M - 0.161) ** 2) * Abase / S_ref
    
    def CDi(self, C_L, AR, e_factor):
        return ((C_L) ** 2) / (np.pi * AR * e_factor)
    
    def Cd_w(self, C_L, SweepLE, IF_w):
        return IF_w * Cd(C_L / (np.cos(SweepLE) ** 2))
    
    def CD(self, CD0, CDi, CD_base, CD_upsweep, Cd_w):
        return CD0 + CDi + CD_base + CD_upsweep + Cd_w
    
    def Drag(self, CD, rho, V, S_ref):
        return CD * 0.5 * rho * (V ** 2) * S_ref
    
    def Drag_polar(self, CD0, CD_base, CD_upsweep, AR, e_factor):
        CDmin = CD0 + CD_base + CD_upsweep
        K = 1 / (np.pi * AR * e_factor)
        return CDmin, K
    
    def CL_des(self, C_L_lst, CD):
        LD = C_L_lst / CD
        index = np.where(LD == np.max(LD))
        return float(C_L_lst[index]), np.max(LD)