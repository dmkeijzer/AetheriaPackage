from dataclasses import dataclass

@dataclass
class Fuselage():
    cockpit_parabol_length: float 
    cylindrical_fuselage_length: float
    conical

@dataclass
class FlightParam():
    mach_cruise: float
    rho_cruise: float
    viscosity: float
    V_cruise: float
    V_hover: float
    V_stall: float
    V_max: float

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


class componentdrag:
    def __init__(self, type, S_ref, l1, l2, l3, d, V_cr, rho, MAC, AR1, AR2, M_cr, k, frac_lam_f, frac_lam_w, mu, tc,xcm,sweepm, sweepLE, u, c_t,h, IF_f, IF_w,IF_v, C_L_minD, Abase, S_v,s1,s2, h_wl1,h_wl2 , k_wl):
        self.S_ref = S_ref
        self.l1 = l1 #parabolic cockpit section
        self.l2 = l2    #cylindrical fuselage length 
        self.l3 = l3 # conical tail length 
        self.d = d #fuselage diameter
        self.V = V_cr
        self.rho = rho 
        self.c = MAC #mac
        self.b = np.sqrt((0.5*(s1*AR1+s2*AR2))*S_ref)
        self.h_wl1 = h_wl1 #winglet height 1
        self.h_wl2 = h_wl2 #winglet height 2
        self.AR = 0.5*(s1*(AR1*winglet_factor(h_wl1, np.sqrt(AR1*S_ref*s1), k_wl))+ s2*(AR2*winglet_factor(self.h_wl2, np.sqrt(AR2*S_ref*s2), k_wl)))
        #0.5*(s1*(AR1+winglet_dAR(AR1,self.h_wl1, np.sqrt(AR1*S_ref*s1)))+ s2*(AR2+winglet_dAR(AR2,self.h_wl2, np.sqrt(AR2*S_ref*s2))))
        self.e = e #oswald efficiency factor
        self.M = M_cr #cruise mac number
        self.k = k # surfaces friction value of zoiets
        self.frac_lamf = frac_lam_f #fraction laminer flow fuselage
        self.frac_lamw = frac_lam_w #fraction laminer flow wing
        self.mu = mu #viscosity
        self.l = self.l1+self.l2+self.l3 #fuselage length 
        self.toc = tc #t/c
        self.xcm = xcm #position maximum thickness wing
        self.sweepm = sweepm 
        self.u = u #upsweep angle [rad]
        self.type = type
        self.IF_v = IF_v #interference factor vertical tail
        self.IF_w = IF_w #interference factor wing
        self.IF_f = IF_f #interference factor 
        self.Abase = Abase #fuselage base area
        if self.type == 'box':
            self.S_c = c_t*h
        self.S_v = S_v  #surface area vertical tail
        self.S_t = 0.5*(1.4*c_t)*h_wl1*2+0.5*(1.4*c_t)*h_wl2*2
        self.SweepLE = sweepLE #sweep LE
        self.C_L_minD = C_L_minD / (np.cos(self.SweepLE) ** 2) #cl at minimum cd
        self.h = h #height between the wings
        self.k_wl = k_wl #smoothiness factor winglet

    def e_OS(self):
        AR = [4,6,8,10]
        e = [0.997, 0.993,0.990, 0.98]
        curve = interp1d(AR, e)
        if self.AR <4:

            return 0.997
        else:
            return curve(self.AR)
    def e_factor(self):
        """
        h = height difference between wings
        b = span
        """
        ratio = self.h / self.b
        if self.type == 'box':
            return self.e_OS() * (0.44 + ratio * 2.219) / (0.44 + ratio * 0.9594)
        if self.type == 'tandem':
            factor = 0.5 + (1 - 0.66 * ratio) / (2.1 + 7.4 * ratio)
            return self.e_OS() * factor ** (-1)
        if self.type == 'normal':
            return (1 + (1-0.66*ratio)/(1.05+3.7*ratio))**(-1)
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
        # print("CD0_fus", self.CD0_f*1.05)

        return CD0

    def CD_upsweep(self):

        return 3.83*(self.u**2.5)*np.pi*self.d**2/(4*self.S_ref)

    def CD_base(self):

        return (0.139 + 0.419*(self.M-0.161)**2) * self.Abase/(self.S_ref)

    def CDi(self, C_L):
        #print(C_L, self.AR, self.e_factor())
        return ((C_L)**2)/(np.pi *self.AR *self.e_factor())

    def Cd_w(self, C_L):

        return self.IF_w*Cd(C_L/(np.cos(self.SweepLE)**2))

    def CD(self, C_L):

        return self.CD0()+self.CDi(C_L) + self.CD_base() +self.CD_upsweep() + self.Cd_w(C_L)

    def Drag(self, C_L):

        return self.CD(C_L) *0.5*self.rho* (self.V**2)*self.S_ref

    def Drag_polar(self):
        CDmin = self.CD0()+ self.CD_base() +self.CD_upsweep()
        K = 1/(np.pi*self.AR*self.e_factor())
        return CDmin, K

    def CL_des(self):
        C_L_lst = np.arange(0,1.5,0.001)
        LD = C_L_lst/self.CD(C_L_lst)
        index = np.where(LD==np.max(LD))
        return float(C_L_lst[index]), np.max(LD)

