import numpy as np
from numpy import pi
from MathFunctions import StepFunction
import pandas as pd

class StructuralError(Exception):
    def __init__(self, msg):
        super().__init__(msg)

class Stringer:
    def __init__(self, area = 0.001, point = (0, 0), **kwargs):
        self.a, (self.x, self.y) = area, point
        
    __repr__ = __str__ = lambda self: f"Stringer(Area={str(self.a)}, Position={str(self.x), str(self.y)})"
    
    Ixx = lambda self: self.a * self.y**2
    Iyy = lambda self: self.a * self.x**2

class ZStringer(Stringer):
    def __init__(self, point = (0, 0), bflange = 0.05, tflange = 0.05, vflange = 0.05, tstr = 0.001):
        super().__init__((tflange + bflange1 + bflange2 + 2*vflange)*tstr, point)
        self.bflange, self.tflange, self.vflange, self.t = bflange, tflange, vflange, tstr

        self.a = (self.bflange + self.tflange + self.vflange) * self.t
    ccarea = lambda self: self.a
    __repr__ = __str__ = lambda self: "Z-Stringer(" + ', '.join(f"{k}={self.__dict__[k]}" for k in self.__dict__) + ")"
    
    def cripplingStress(self, E, v, sigma_y):
        alpha, n = 0.8, 0.6
        sdict = {}
        sdict['ccstress1'] = 0.8 * (0.425/sigma_y * (np.pi**2 * E / (12 * (1 - v**2))) * (self.t / self.bflange)**2) ** (1 - n) * sigma_y
        sdict['ccstress2'] = 0.8 * (0.425/sigma_y * (np.pi**2 * E / (12 * (1 - v**2))) * (self.t / self.tflange)**2) ** (1 - n) * sigma_y
        sdict['ccstress3'] = 0.8 * (4/sigma_y * (np.pi**2 * E / (12 * (1 - v**2))) * (self.t / self.vflange)**2) ** (1 - n) * sigma_y
        for (key, value) in sdict.items():
            if value > sigma_y:
                sdict[key] = sigma_y
        return (sdict['ccstress1'] * self.t * self.bflange 
                + sdict['ccstress2'] * self.t * self.tflange
                + sdict['ccstress3'] * self.t * self.vflange) / self.a
    
class HatStringer(Stringer):
    def __init__(self, point = (0, 0), bflange1 = 0.05, bflange2 =0.05, tflange = 0.05, vflange = 0.05, tstr = 0.001):
        super().__init__((tflange + bflange1 + bflange2 + 2*vflange)*tstr, point)
        self.bflange1, self.bflange2, self.tflange, self.vflange, self.t = bflange1, bflange2, tflange, vflange, tstr
        self.a = (self.bflange1 + self.bflange2 + self.tflange + 2*self.vflange) * self.t
        
    ccarea = lambda self: self.a
    __repr__ = __str__ = lambda self: "HAT-Stringer(" + ', '.join(f"{k}={self.__dict__[k]}" for k in self.__dict__) + ")"
    
    
    def cripplingStress(self, E, v, sigma_y):
        alpha, n = 0.8, 0.6
        sdict = {}
        sdict['ccstress1'] = 0.8 * (0.425/sigma_y * (np.pi**2 * E / (12 * (1 - v**2))) * (self.t / (self.bflange1))**2) ** (1 - n) * sigma_y
        sdict['ccstress5'] = 0.8 * (0.425/sigma_y * (np.pi**2 * E / (12 * (1 - v**2))) * (self.t / (self.bflange2))**2) ** (1 - n) * sigma_y
        sdict['ccstress24'] = 0.8 * (4/sigma_y * (np.pi**2 * E / (12 * (1 - v**2))) * (self.t / (self.vflange))**2) ** (1 - n) * sigma_y
        sdict['ccstress3'] = 0.8 * (4/sigma_y * (np.pi**2 * E / (12 * (1 - v**2))) * (self.t / (self.tflange))**2) ** (1 - n) * sigma_y
        for (key, value) in sdict.items():
            if value > sigma_y:
                sdict[key] = sigma_y
        return (sdict['ccstress1'] * self.t * self.bflange1
                + sdict['ccstress5'] * self.t * self.bflange2
                + 2 * sdict['ccstress24'] * self.t * self.vflange + sdict['ccstress3'] * self.t * self.tflange) / self.a

class JStringer(Stringer):
    def __init__(self, point = (0, 0), bflange1 = 0.05, bflange2 =0.05, tflange = 0.05, vflange = 0.05, tstr = 0.001):
        super().__init__((tflange + bflange1 + bflange2 + 2*vflange)*tstr, point)
        self.bflange1, self.bflange2, self.tflange, self.vflange, self.t = bflange1, bflange2, tflange, vflange, tstr
        
        self.a = (self.bflange1 + self.bflange2 + self.tflange + self.vflange) * self.t
        
    ccarea = lambda self: self.a
    __repr__ = __str__ = lambda self: "J-Stringer(" + ', '.join(f"{k}={self.__dict__[k]}" for k in self.__dict__) + ")"
    
    def cripplingStress(self, E, v, sigma_y):
        alpha, n = 0.8, 0.6
        self.sdict = {}
        self.sdict['ccstress1'] = 0.8 * (0.425/sigma_y * (np.pi**2 * E / (12 * (1 - v**2))) * (self.t / (self.bflange1))**2) ** (1 - n) * sigma_y
        self.sdict['ccstress2'] = 0.8 * (0.425/sigma_y * (np.pi**2 * E / (12 * (1 - v**2))) * (self.t / (self.bflange2))**2) ** (1 - n) * sigma_y
        self.sdict['ccstress3'] = 0.8 * (4/sigma_y * (np.pi**2 * E / (12 * (1 - v**2))) * (self.t / (self.vflange))**2) ** (1 - n) * sigma_y
        self.sdict['ccstress4'] = 0.8 * (0.425/sigma_y * (np.pi**2 * E / (12 * (1 - v**2))) * (self.t / (self.tflange))**2) ** (1 - n) * sigma_y
        for (key, value) in self.sdict.items():
            if value > sigma_y:
                self.sdict[key] = sigma_y
        return (self.sdict['ccstress1'] * self.t * self.bflange1
                + self.sdict['ccstress2'] * self.t * self.bflange2
                + self.sdict['ccstress3'] * self.t * self.vflange + self.sdict['ccstress4'] * self.t * self.tflange) / self.a
    
class WingBox:
    def __init__(self, thicknessOfSkin, thicknessOfSpar, base, height, stringers = []):
        self.b, self.h, self.tsk, self.tsp = base, height, thicknessOfSkin, thicknessOfSpar
        self.str = stringers
        tstrs, bstrs = sum(1 for stringer in stringers if stringer.y > 0), sum(1 for stringer in stringers if stringer.y < 0)
        self.tspitch, self.bspitch = self.b/(tstrs + 1), self.b/(bstrs + 1)

    
    __str__ = __repr__ = lambda self: \
    f"Wingbox(Height={str(self.h)}, Base={str(self.b)}, Tsk = {str(self.tsk)}, Tsp = {str(self.tsp)}, Stringers = {len(self.str)})"
    
    Area = lambda self: self.b * self.h - (self.b - 2 * self.tsp) * (self.h - 2 * self.tsk) + sum(s.a for s in self.str)

    Ixx = lambda self: (self.tsp * self.h ** 3 + self.b * self.tsk ** 3) / 6 + (self.tsk * self.b * self.h ** 2) / 2 + sum(s.Ixx() for s in self.str)
    
    Iyy = lambda self: (self.tsk ** 3 * self.h + self.b ** 3 * self.tsp) / 6 + (self.tsp * self.h * self.b ** 2) / 2 + sum(s.Iyy() for s in self.str)
    
    Vc = Ixy = lambda self: 0
    
    def StrPlacement(self, nstr_top:int , nstr_bottom:int, stringerGeometry = {}, stringerType = 'Point'):
        strtype = {'Z':ZStringer, 'Hat':HatStringer, 'J':JStringer, 'Point': Stringer}[stringerType]
        topstringers = [strtype(point = (i*self.b/(nstr_top + 1) - self.b/2, self.h/2), **stringerGeometry) for i in range(1, nstr_top+1)]
        bottomstringers = [strtype(point = (i*self.b/(nstr_bottom + 1) - self.b/2, -self.h/2), **stringerGeometry) for i in range(1, nstr_bottom+1)]
        self.tspitch, self.bspitch = self.b/(nstr_top + 1), self.b/(nstr_bottom + 1)
        self.str = self.str.copy() + topstringers + bottomstringers
        
    def Vshear(self, Vy, x, y):
        Ixx = self.Ixx()
        inrge = lambda l1, u1, l2, u2: l1 <= x <= u1 and l2 <= y <= u2
        vit = - Vy * self.tsp / Ixx if (-self.b/2 <= x <= -self.b/2 + self.tsp) or (self.b/2 - self.tsp <= x <= self.b/2) else - Vy * self.tsk / Ixx
        if inrge(0, self.b/2 - 1.5*self.tsp, -self.h/2, -self.h/2 + self.tsk):
            return vit * (-self.h * x / 2)
        elif inrge(self.b/2-1.5*self.tsp, self.b/2, -self.h/2, self.h/2):
            s = self.h/2 + y
            return vit * (0.5 * s * s - self.h * s / 2) + self.Vshear(Vy, self.b/2 - 1.5*self.tsp, -self.h/2)
        elif inrge(-self.b/2 + 1.5*self.tsp, self.b/2 - 1.5*self.tsp, self.h/2-self.tsk, self.h/2):
            s = self.b/2 - x
            return vit * (self.h*s/2) + self.Vshear(Vy, self.b/2, self.h/2)
        elif inrge(-self.b/2, -self.b/2+1.5*self.tsp, -self.h/2, self.h/2):
            s = self.h/2 - y
            return vit * (-0.5 * s * s + self.h * s / 2) + self.Vshear(Vy, -self.b/2 + 1.5*self.tsp, self.h/2)
        elif inrge(-self.b/2 + 1.5*self.tsp, 0, -self.h/2, -self.h/2+self.tsk):
            return vit * (-self.h * (x + self.b/2) / 2) + self.Vshear(Vy, -self.b/2, -self.h/2)
        else:
            raise ValueError(f"Invalid Coordinates Supplied: {(x, y) = }")

    def Hshear(self, Vx, x, y):
        Iyy = self.Iyy()
        inrge = lambda l1, u1, l2, u2: l1 <= x <= u1 and l2 <= y <= u2
        vit = - Vx * self.tsp / Iyy if (-self.b/2 <= x <= -self.b/2 + self.tsp) or (self.b/2 - self.tsp <= x <= self.b/2) else - Vx * self.tsk / Iyy
        if inrge(-self.b/2, -self.b/2+self.tsp, -self.h/2, 0):
            return vit * (self.b * y / 2)
        elif inrge(-self.b/2, self.b/2, -self.h/2, -self.h/2+self.tsk):
            s = x + self.b/2
            return vit * (0.5 * s * s - self.b * s / 2) + self.Hshear(Vx, -self.b/2, -self.h/2)
        elif inrge(self.b/2 -self.tsp, self.b, -self.h/2, self.h/2):
            return vit * (self.b * (y + self.h/2) / 2) + self.Hshear(Vx, self.b/2, -self.h/2)
        elif inrge(-self.b/2, self.b/2, self.h/2-self.tsk, self.h/2):
            s = -x + self.b/2
            return vit * (-0.5 * s * s + self.b * s / 2) + self.Hshear(Vx, self.b/2, self.h/2)
        elif inrge(-self.b/2, -self.b/2+self.tsp, 0, self.h/2):
            return vit * (-self.b * (self.h/2-y) / 2) + self.Hshear(Vx, -self.b/2, self.h/2)
        else:
            raise ValueError(f"Invalid Coordinates Supplied: {(x, y) = }")
    
    q = lambda self, x, y, Vx=0, Vy=0, T=0: self.Vshear(Vy, x, y) + self.Hshear(Vx, x, y) + T / (2 * self.Area())
    tau = lambda self, x, y, Vx=0, Vy=0, T=0: self.q(x, y, Vx, Vy, T) / (self.tsp if \
        (-self.b/2 <= x <= -self.b/2 + self.tsp) or (self.b/2 - self.tsp <= x <= self.b/2) else self.tsk)
    
    o = lambda self, x, y, Mx=0, My=0: My * x / self.Iyy() + Mx * y / self.Ixx()
    
    def Bstress(self, EofStringers, vOfStringers, yieldOfStringers, EofSkin, vOfSkin, top_panel = True):
        # stringer properties - in case different material is used
        Estr, vstr, ystr = EofStringers, vOfStringers, yieldOfStringers
        ccstr = self.str[0].cripplingStress(Estr, vstr, ystr)
        ccarea = self.str[0].ccarea()
        # skin properties
        Esk, vsk = EofSkin, vOfSkin
        # pitch depends on which panel is taken, top or bottom
        pitch = self.tspitch if top_panel else self.bspitch
        sigma_crskin = 4 * (np.pi ** 2 * Esk/(12 * (1 - vsk**2)))*(self.tsk / (pitch))**2
        C = 6.98 if pitch/self.tsk > 75 else 4
        we = 0 #self.tsk * np.sqrt(np.pi ** 2 * C * Esk/(ccstr*12*(1-vsk**2)))
        new_pitch = pitch - we
        new_sigma_crskin = 4 * (np.pi ** 2 * Esk/(12 * (1 - vsk**2)))*(self.tsk / (new_pitch))**2
        if new_pitch < 0:
            raise StructuralError("Invalid pitch length: " + str(new_pitch))
        return (new_sigma_crskin * new_pitch * self.tsk + ccstr * (ccarea + we * self.tsk))/ (new_pitch * self.tsk + (ccarea + we * self.tsk))

class WingStructure:
    def __init__(self, span, taper, rootchord, wingbox):
        self.span, self.taper, self.rc = span, taper, rootchord
        self.tc = self.rc * self.taper
        self.box = wingbox
    
    chord = lambda self, z: self.rc * (self.taper - 1) * z / (self.span/2) + self.rc if (0 <= z <= self.span/2) else None
    __call__ = lambda self, z: WingBox(self.box.tsk, self.box.tsp, self.box.b*self.chord(z), self.box.h*self.chord(z), self.box.str)
    __repr__ = __str__ = lambda self: "WingStructure(" + ', '.join(f"{k}={self.__dict__[k]}" for k in self.__dict__) + ")"
    area = lambda self: 0.5 * self.rc * (1 + self.taper) * self.span
