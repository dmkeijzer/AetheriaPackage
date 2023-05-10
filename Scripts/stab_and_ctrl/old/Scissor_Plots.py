import numpy as np
from scipy.linalg import null_space
from matplotlib import pyplot as plt
from matplotlib import colors as mc
from stab_and_ctrl.hover_controllabilty import HoverControlCalcTandem
import constants as consts
from Aero_tools import ISA


class Wing_placement_sizing:
    def __init__(self, W,h, lfus, hfus, wfus, V0, CD0, CLfwd,
                 CLrear,CLdesfwd,CLdesrear,Clafwd,Clarear, Cmacfwd, Cmacrear, Sfwd, Srear,
                 Afwd, Arear, Gamma, Lambda_c4_fwd, Lambda_c4_rear, cfwd,
                 crear, bfwd, brear, efwd, erear, taper, n_rot_f, n_rot_r,
                 rot_y_range_f, rot_y_range_r, K, ku,Zcg,d,dy,Pbr,Vr_V2):

        self.W = W         # Weight [N]
        self.h = h       # Height [m]
        Aero = ISA(self.h) # Imports ISA class
        self.rho = Aero.density() #density [kg/m^3]
        self.T = Aero.temperature() # Temperature [K]
        # self.CLdesfwd,self.CLdesrear = CLdesfwd,CLdesrear # DESIGN CL
        self.Pbr = Pbr # Shaft power of one engine [W]
        self.lfus = lfus # Length of the fuselage
        self.hfus = hfus # Height of the fuselage [m]
        self.wfus = wfus # Width of the fuselage [m]
        self.Srear = Srear # Rear wing area [m^2]
        self.Sfwd = Sfwd   # Forward wing area [m^2]
        self.S = Srear+Sfwd # Aircraft wing area [m^2]
        self.cfwd = cfwd         # Average chord [m]
        self.crear = crear  # Average chord [m]
        self.bfwd = bfwd         # Wing span [m]
        self.brear = brear # Wing span [m]
        self.efwd, self.erear = efwd,erear
        self.taper = taper
        self.Afwd, self.Arear = Afwd, Arear # Aspect ratio of both wings [-]
        self.Sweepc4fwd = Lambda_c4_fwd # Sweep at c/4 [rad]
        self.Sweepc4rear = Lambda_c4_rear # Sweep at c/4 [rad]
        self.Sweepc2fwd = self.Sweep(Afwd,self.Sweepc4fwd,50,25)
        self.Sweepc2rear = self.Sweep(Arear, self.Sweepc4rear, 50, 25)
        # print("Sweep at c/2:",self.Sweepc2fwd*180/np.pi)
        self.V0 = V0       # Initial speed [m/s]
        self.M0 = self.V0/(1.4*287*self.T)       # Initial mach number [-]
        self.Gamma_fwd = Gamma # Forward wing dihedral [rad]
        self.CLfwd = CLfwd # MAX Forward lift coefficient [-]
        self.CLrear = CLrear # MAX Rear lift coefficient [-]
        self.Clafwd,self.Clarear = Clafwd,Clarear # Airfoil lift slopes [1/rad]
        # self.CLafwd, self.CLarear = CLafwd, CLarear # Wing lift curve slopes for both wings [1/rad]
        self.CLafwd = self.C_L_a(self.Afwd,self.Sweepc2fwd)# Wing lift curve slopes for both wings [1/rad]
        self.CLarear = self.C_L_a(self.Arear,self.Sweepc2rear)# Wing lift curve slopes for both wings [1/rad]
        # print("CLafwd,CLarear = ",self.CLafwd,self.CLarear)
        # print("Sweep half chord = ", self.Sweepc2fwd*180/np.pi)
        self.Cmacfwd, self.Cmacrear = Cmacfwd,Cmacrear
        self.CD0 = CD0 # C_D_0 of forward wing
        self.d = d  # Horizontal distance alteration for forward wing [m]
        self.dy = dy # Vertical distance alteration for forward wing [m]
        self.xacfwd = 0.25*self.cfwd + d
        self.xacrear = self.lfus - (1 - 0.25) * self.crear
        # self.CLdesfwd = self.W / (0.5 * self.rho * self.V0 ** 2 * self.Sfwd) / 2
        # self.CLdesrear = self.W / 2 / (0.5 * self.V0 ** 2 * self.rho * self.Srear)
        self.CLdesfwd = CLdesfwd
        self.CLdesrear = CLdesrear
        self.de_da = self.deps_da(self.Sweepc4fwd,self.bfwd,self.lh(),self.hfus-self.dy,self.Afwd,self.CLafwd)
        # print("de/da = %.3f"%(self.de_da))
        self.Vr_V2 = Vr_V2
        self.Zcg = Zcg
        self.hover_calc = HoverControlCalcTandem(W / consts.g, n_rot_f,
                                                 n_rot_r, self.xacfwd,
                                                 self.xacrear, rot_y_range_f,
                                                 rot_y_range_r, K, ku)
    def lh(self):
        return abs(self.xacfwd-self.d - self.xacrear)

    def deps_da(self,Lambda_quarter_chord, b, lh, h_ht, A, CLaw):
        """
        Inputs:
        :param Lambda_quarter_chord: Sweep Angle at c/4 [RAD]
        :param lh: distance between ac_w1 with ac_w2 (horizontal)
        :param h_ht: distance between ac_w1 with ac_w2 (vertical)
        :param A: Aspect Ratio of wing
        :param CLaw: Wing Lift curve slope
        :return: de/dalpha
        """
        r = lh * 2 / b
        mtv = h_ht * 2 / b # Approximation
        Keps = (0.1124 + 0.1265 * Lambda_quarter_chord + 0.1766 * Lambda_quarter_chord ** 2) / r ** 2 + 0.1024 / r + 2
        Keps0 = 0.1124 / r ** 2 + 0.1024 / r + 2
        v = 1 + (r ** 2 / (r ** 2 + 0.7915 + 5.0734 * mtv ** 2)) ** (0.3113)
        de_da = Keps / Keps0 * CLaw / (np.pi * A) * (
                r / (r ** 2 + mtv ** 2) * 0.4876 / (np.sqrt(r ** 2 + 0.6319 + mtv ** 2)) + v * (
                1 - np.sqrt(mtv ** 2 / (1 + mtv ** 2))))
        phi = np.arcsin(mtv/r)
        # print("r, mtv = ",r,mtv)
        # print("phi = %.3f"%(phi*180/np.pi))
        # if 180/np.pi*phi<30 and 180/np.pi*phi>0: #To account for propeller downwash
        #     dsde_da = 6.5*(self.rho*self.Pbr**2*self.Sfwd**3*self.CLdesfwd**3/(self.lh()**4*self.W**3))**(1/4)*(np.sin(phi*6))**2.5
        # else:
        #     dsde_da = 0
        Sf = b**2/A
        dsde_da = np.where(np.logical_and(np.rad2deg(phi) < 30, np.rad2deg(phi) > 0),
            6.5 * (self.rho * self.Pbr ** 2 * Sf ** 3 * self.CLdesfwd ** 3 / (lh ** 4 * self.W ** 3)) ** (1 / 4) * (np.sin(phi * 6)) ** 2.5,
            0)
        # print("Configuration %.0f de/da = %.4f "%(conf,de_da))
        return de_da+dsde_da

    def Sweep(self,AR,Sweepm,n,m):
        """
        Inputs
        :param AR: Aspect Ratio
        :param Sweepm: Sweep at mth chord [rad]
        :param n: (example quarter chord: n =25)
        :param m: mth chord (example half chord: m=50)
        :return: Sweep at nth chord [rad]
        """
        tanSweep_m = np.tan(Sweepm)
        tanSweep_n = tanSweep_m -4/(AR)*(n-m)/100*(1-self.taper)/(1+self.taper)
        return np.arctan(tanSweep_n)

    def C_L_a(self,A, Lambda_half, eta=0.95):
        """
        Inputs:
        :param b: wing span
        :param S: wing area
        :param Lambda_half: Sweep angle at 0.5*chord
        :param eta: =0.95
        :return: Lift curve slope for tail AND wing using DATCOM method
        """
        M = self.M0
        beta = np.sqrt(1 - M ** 2)
        value =  self.Clafwd* A / (2 + np.sqrt(4 + ((A * beta / eta) ** 2) * (1 + (np.tan(Lambda_half) / beta) ** 2)))
        return value

    def Sr_Sfwd(self, Xcg, elevator, d):
        """
        Inputs:
        :param Xcg: Array of cg positions from 1 to the length of the fuselage
        :param elevator: Elevator effectiveness factor to CL_fwd
        :param d: Alterning parameter d for position change of rear wing
        :return: S_fwd/S_rear array for both stability and controllability
        """
        CLfwd = self.CLfwd * elevator
        CDfwd = self.CD0 + self.CLfwd ** 2 / (np.pi * self.Afwd * self.efwd)
        CDrear = self.CD0 + self.CLrear ** 2 / (np.pi * self.Arear * self.erear)
        c = self.Sfwd / (self.Sfwd + self.Srear) * self.cfwd + self.Srear / (self.Srear + self.Sfwd) * self.crear
        CDafwd = 2*self.CLafwd*self.CLdesfwd/(np.pi*self.Afwd*self.efwd)
        CDarear = 2*self.CLarear*self.CLdesrear/(np.pi*self.Arear*self.erear)
        # print("CD_alpha = ",CDafwd)
        deda = self.de_da
        # print("de/da = ",deda)
        SrSfwd_stab = (-self.CLafwd * (Xcg - self.xacfwd) +CDafwd*(self.Zcg-self.dy))/ \
                      (-self.CLarear * (1 - deda) * (self.xacrear  - Xcg)+CDarear*(self.hfus-self.Zcg)*(1-deda))
        SrSfwd_control = (-self.Cmacfwd * self.cfwd + CDfwd * (self.Zcg-self.dy) - CLfwd * (Xcg - self.xacfwd) / (
                        CDrear * (self.hfus-self.Zcg) - self.CLrear * (self.xacrear - Xcg) + self.Cmacrear * self.crear))
        return SrSfwd_stab ** (-1), SrSfwd_control ** (-1)

    def plotting(self, x_min, x_max, dx, elevator, d, n_failures=2, y_cg=0):
        Sr_Sfwd = np.linspace(0, 2.5, 100)
        Sforward = self.S*(1/(1+Sr_Sfwd))
        Sr = self.S- Sforward
        cforward = self.cfwd*np.sqrt(Sforward/self.Afwd)/np.sqrt(self.Sfwd/self.Afwd)
        cr = self.crear*np.sqrt(Sr/self.Arear)/np.sqrt(self.Srear/self.Arear)
        xacfwd_1 = 0.25*cforward + self.d
        xacrear_1 = self.lfus - (1 - 0.25) * cr
        # xacfwd_1 = self.xacfwd
        # xacrear_1 = self.xacrear
        # CLfwd = self.CLfwd * elevator
        CDfwd = self.CD0 + self.CLfwd ** 2 / (np.pi * self.Afwd * self.efwd)
        CDrear = self.CD0 + self.CLrear ** 2 / (np.pi * self.Arear * self.erear)
        # c = self.Sfwd / (self.Sfwd + self.Srear) * self.cfwd + self.Srear / (self.Srear + self.Sfwd) * self.crear
        CDafwd = 2 * self.CLafwd * self.CLdesfwd / (np.pi * self.Afwd * self.efwd)
        CDarear = 2 * self.CLarear * self.CLdesrear / (np.pi * self.Arear * self.erear)
        # print("CD_alpha = ",CDafwd)
        deda = self.deps_da(self.Sweepc4fwd,np.sqrt(Sforward*self.Afwd),xacrear_1-xacfwd_1,self.hfus-self.dy,self.Afwd,self.CLafwd)
        o = -self.Cmacfwd*cforward+ CDfwd*(self.Zcg-self.dy)+self.CLfwd*elevator*xacfwd_1-\
            self.Vr_V2*(CDrear*(self.hfus-self.Zcg)*Sr_Sfwd-self.CLrear*Sr_Sfwd*xacrear_1+self.Cmacrear*Sr_Sfwd*cr)
        p = self.CLfwd*elevator+self.Vr_V2*self.CLrear*Sr_Sfwd
        xcg_control = o/p

        oo = CDafwd*(self.Zcg-self.dy)+self.CLafwd*xacfwd_1-\
            self.Vr_V2*(CDarear*(self.hfus-self.Zcg)*(1-deda)*Sr_Sfwd-self.CLarear*Sr_Sfwd*xacrear_1*(1-deda))
        pp = self.CLafwd + self.Vr_V2 * self.CLarear * Sr_Sfwd*(1-deda)
        xcg_stability = oo/pp
        Xcg = np.arange(x_min, x_max, dx)
        SrSfwd_stability = self.Sr_Sfwd(Xcg,elevator,d)[0]
        SrSfwd_control = self.Sr_Sfwd(Xcg, elevator, d)[1]
        plt.plot(Sr_Sfwd,xcg_stability,label="Neutral Stability Line")
        plt.plot(Sr_Sfwd,xcg_control,label="Controllability Line")

        # x_hover_min, x_hover_max, hover_fail_limit_front, hover_fail_limit_aft = self.hover_calc.calc_crit_x_cg_range(x_min, x_max, dx, y_cg, [(self.xacfwd + self.xacrear) / 2, 0], [n_failures])[0]
        # plt.axvline(x_hover_min, label="Hover Controllability Limits for " + str(n_failures) + " Engine Failures", color="tab:red")
        # plt.axvline(x_hover_max, color="tab:red")

        plt.ylabel(r"$x_{cg}$ [m]",fontsize=14)
        plt.xlabel(r"$\frac{S_{rear}}{S_{fwd}}$ [-]",fontsize=14)
        # plt.ylim(0.5, 1.5)
        plt.legend()

        plt.show()
