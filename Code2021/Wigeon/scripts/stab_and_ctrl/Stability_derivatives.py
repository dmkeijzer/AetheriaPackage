import numpy as np
from scipy.linalg import null_space
from matplotlib import pyplot as plt
from matplotlib import colors as mc
from Aero_tools import ISA
from stab_and_ctrl.Aileron_Sizing import Control_surface
from stab_and_ctrl.Vertical_tail_sizing import VT_sizing as vertical_tail

"""
Novel analytical model to obtain stability derivatives for a tandem wing configuration. 
author: Michal Cuadrat-Grzybowski
"""

class Stab_Derivatives:
    def __init__(self,W,h,lfus,hfus,wfus, d,dy,xcg,zcg,cfwd,crear,Afwd,Arear,Vstall,
                 V0,T0,CLfwd0,CLrear0,CD0,CL0,theta0,alpha0,
                 Clafwd,Clarear, Cd0fwd, Cd0rear, CLafwd,CLarear,Sfwd,Srear,Gamma_fwd,Gamma_rear,
                 efwd,erear,Lambda_c4_fwd,Lambda_c4_rear,taper,taper_v,
                 bv,Sv,ARv,sweepTE, Pbr,C_D_0,eta_rear,eta_v):
        self.W = W         # Weight [N]
        self.h = h     # Height [m]
        Aero = ISA(self.h) # Initialises Aero object
        self.rho = Aero.density() # Density [kg/m^3]
        self.T = Aero.temperature() # Temperature [K]
        self.lfus = lfus  # Length of the fuselage
        self.hfus = hfus  # Height of the fuselage [m]
        self.wfus = wfus  # Maximum width of the fuselage [m]
        self.Srear = Srear # Rear wing area [m^2]
        self.Sfwd = Sfwd   # Forward wing area [m^2]
        self.S = Srear+Sfwd # Aircraft wing area [m^2]
        self.Afwd,self.Arear  =Afwd,Arear # Aspect ratio of both wings [-]
        self.cfwd = cfwd         # Average chord [m]
        self.crear = crear  # Average chord [m]
        self.c = self.cfwd*self.Sfwd/(self.S) + self.crear*self.Srear/(self.S)
        self.bfwd = np.sqrt(self.Afwd*self.Sfwd)     # Wing span [m]
        self.brear = np.sqrt(self.Arear*self.Srear) # Wing span [m]
        self.Sweepc4 = Lambda_c4_fwd # Sweep at c/4 [rad]
        self.Sweepc4_rear = Lambda_c4_rear # Sweep at c/4 [rad]
        self.taper = taper # Wing taper ratio [-]
        self.taper_v = taper_v # Vertical tail taper ratio [-]
        self.efwd,self.erear = efwd, erear # Span efficiency factors of both wings
        self.bv = bv       # Vertical tail span [m]
        self.Sv = Sv       # Vertical tail area [m^2]
        self.xcg = xcg  # CG longitudinal position wrt Nose [m]
        self.th0 = theta0  # Initial pitch angle [rad]
        self.alpha0 = alpha0 # Initial AoA [rad]
        self.V0 = V0       # Initial speed [m/s]
        self.M0 = V0/(np.sqrt(1.4*287*self.T)) # Initial mach number [-]
        self.CL0 = CL0     # Initial lift coefficient [-]
        self.CLfwd0 = CLfwd0 # Initial lift coefficient of fwd wing [-]
        self.CLrear0 = CLrear0 # Initial lift coefficient of rear wing [-]
        self.CD0 = CD0     # Initial total drag coefficient (not profile CD_0)[-]
        self.T0 = T0       # Initial thrust [N]
        self.Gamma_fwd = Gamma_fwd # Forward wing dihedral [rad]
        self.Gamma_rear = Gamma_rear  # Rear wing dihedral [rad]
        self.CLafwd, self.CLarear = CLafwd, CLarear # Wing lift curve slopes for both wings [1/rad]
        self.Clafwd, self.Clarear = Clafwd,Clarear # Airfoil lift curve slopes
        self.Cd0fwd,self.Cd0rear = Cd0fwd,Cd0rear # Airfoil base drag
        self.Cl0fwd = CLfwd0/(np.cos(self.Sweep(self.Afwd,self.taper,0,25,0)))**2 # Airfoil lift at zero angle of attack
        self.Cl0rear = CLrear0/(np.cos(self.Sweep(self.Arear, self.taper, 0, 25, 0)))**2 # Airfoil lift at zero angle of attack
        self.d = d
        self.dy = dy
        # self.xacfwd = 0.25 * self.cfwd + self.d
        # self.xacrear = self.lfus - (1 - 0.25) * self.crear
        self.xacfwd = 0.5
        self.xacrear = 6.1
        self.Pbr = Pbr # Shaft power per engine [W]
        self.zcg = zcg
        self.ARv = ARv
        self.b = np.sqrt(0.5*(self.Srear/self.S*self.Arear+self.Sfwd/self.S*self.Afwd)*self.S)
        self.CD_0 = C_D_0 # PROFILE DRAG for one wing [-]
        self.Vstall = Vstall # Stall speed [m/s]
        self.eta_rear = eta_rear
        self.eta_v = eta_v
        self.sweepTE = sweepTE
        VT = vertical_tail(self.W,self.h,self.xcg,self.lfus,self.hfus,self.wfus,self.V0,self.Vstall,
                           self.CD0,self.CLfwd0,self.CLrear0,self.CLafwd,self.CLarear,self.Sfwd,self.Srear,
                           self.Afwd,self.Arear,self.Sweepc4,self.Sweepc4_rear,self.cfwd,self.crear,
                           self.bfwd,self.brear,self.taper,self.ARv,self.sweepTE)
        self.xacv = VT.xacv(self.ARv, self.sweepTE)
        self.lv = self.xacv-self.xcg
        # print("xacv, xcg, lv", self.xacv,self.xcg, self.lv)
        ### It is assumed that aeroelastic effects are neglected ###

    def lh_arm(self):
        return abs(self.xacfwd-self.d - self.xacrear)

    def deps_da(self, Lambda_quarter_chord, h_ht,CLaw,eta=0.5):
        """
        Inputs:
        :param Lambda_quarter_chord: Sweep Angle at c/4 [RAD]
        :param h_ht: distance between ac_w1 with ac_w2 (vertical)
        :param CLaw: Wing Lift curve slope
        :return: de/dalpha
        """
        A = self.Afwd
        b = self.bfwd
        lh = self.lh_arm()
        r = lh * 2 / b
        mtv = h_ht * 2 / b  # Approximation
        Keps = (0.1124 + 0.1265 * Lambda_quarter_chord + 0.1766 * Lambda_quarter_chord ** 2) / r ** 2 + 0.1024 / r + 2
        Keps0 = 0.1124 / r ** 2 + 0.1024 / r + 2
        v = 1 + (r ** 2 / (r ** 2 + 0.7915 + 5.0734 * mtv ** 2)) ** (0.3113)
        de_da = Keps / Keps0 * CLaw / (np.pi * A) * (
                r / (r ** 2 + mtv ** 2) * 0.4876 / (np.sqrt(r ** 2 + 0.6319 + mtv ** 2)) + v * (
                1 - np.sqrt(mtv ** 2 / (1 + mtv ** 2))))
        phi = np.arcsin(mtv / r)
        # print("r, mtv = ",r,mtv)
        # print("phi = %.3f"%(phi*180/np.pi))
        if 180 / np.pi * phi < 30 and 180 / np.pi * phi > 0:  # To account for propeller downwash
            dsde_da = 6.5 * (self.rho * self.Pbr ** 2 * self.Sfwd ** 3 * self.CLfwd0 ** 3 / (
                        lh** 4 * self.W ** 3)) ** (1 / 4) * (np.sin(phi * 6)) ** 2.5
        else:
            dsde_da = 0
        # print("de/da = %.4f, de_P/da =%.4f "%(de_da*eta,dsde_da))
        return de_da*eta + dsde_da
    def Sweep(self,AR,taper,Sweepm,n,m):
        """
        Inputs
        :param AR: Aspect Ratio of VT
        :param Sweepm: Sweep at mth chord [rad]
        :param n: (example quarter chord: n =25)
        :param m: mth chord (example half chord: m=50)
        :return: Sweep at nth chord [rad]
        """
        tanSweep_m = np.tan(Sweepm)
        tanSweep_n = tanSweep_m -4/(AR)*(n-m)/100*(1-taper)/(1+taper)
        return np.arctan(tanSweep_n)

    def C_L_a(self,A, Lambda_half, eta=0.95):
        """
        Inputs:
        :param M: Mach number
        :param b: wing span
        :param S: wing area
        :param Lambda_half: Sweep angle at 0.5*chord
        :param eta: =0.95
        :return: Lift curve slope for tail AND wing using DATCOM method
        """
        M= self.M0
        beta = np.sqrt(1 - M ** 2)
        # print("Lambda_1/2c = ",Lambda_half)
        value = 2 * np.pi * A / (2 + np.sqrt(4 + ((A * beta / eta) ** 2) * (1 + (np.tan(Lambda_half) / beta) ** 2)))
        return value

    def u_derivatives(self):
        """
        Analytical estimates of stability derivatives wrt û = u/V0
        :return: C_X_u, C_Z_u, C_m_u
        """
        # print("M_0 = ",self.M0)
        CT_u = -3*(self.CD0)-3*self.CL0*np.tan(self.th0+self.alpha0)
        CD0fwd = self.CLfwd0**2/(np.pi*self.Afwd*self.efwd) + self.CD_0
        CD0rear = self.CLfwd0**2/(np.pi*self.Afwd*self.efwd) + self.CD_0
        CT_fwd_u = -3*(CD0fwd)-3*self.CLfwd0*np.tan(self.th0+self.alpha0)
        CT_rear_u = -3 * (CD0rear) - 3 * self.CLrear0 * np.tan(self.th0 + self.alpha0)
        CZ_u = (-self.M0**2/(1-self.M0**2)*self.CL0)
        CD_M = 0 # Incompressible flow
        CLfwd_M =  self.M0/(1-self.M0**2)*self.CLfwd0
        CLrear_M  =self.M0/(1-self.M0**2)*self.CLrear0
        CX_u = CT_u-self.M0*CD_M -CZ_u*self.alpha0
        Cm_M = CLfwd_M*(self.xcg-self.xacfwd)*self.Sfwd/(self.S*self.c)-\
               CLrear_M*(self.xacrear-self.xcg)*self.Srear/(self.S*self.c)*self.eta_rear
        Cm_u = self.M0*Cm_M -\
               CT_rear_u*(self.hfus-self.zcg)*self.Srear/(self.S*self.c)*self.eta_rear + \
               CT_fwd_u*(self.zcg-self.dy)*self.Sfwd/(self.S*self.c)
        return CX_u,CZ_u,Cm_u

    def alpha_derivatives(self):
        """
        Analytical estimates of stability derivatives wrt AoA
        :return: C_X_a, C_Z_a, C_m_a
        """
        CDafwd = 2*self.CLafwd*self.CLfwd0/(np.pi*self.Afwd*self.efwd)
        CDarear = 2 * self.CLarear * self.CLrear0 / (np.pi * self.Arear * self.erear)
        de_da = self.deps_da(self.Sweepc4,self.hfus-self.dy,self.CLafwd)
        CDa = CDafwd*self.Sfwd/self.S + CDarear*(1-de_da)*self.Srear/self.S*self.eta_rear
        CLa = self.CLafwd*self.Sfwd/self.S + self.CLarear*(1-de_da)*self.Srear/self.S*self.eta_rear
        CX_a = -CDa + CLa*self.alpha0 + self.CL0
        CZ_a = -CDa*self.alpha0-self.CD0 -CLa
        Cm_a = -CDafwd*(self.zcg-self.dy)*self.Sfwd/(self.S*self.c)+\
               self.CLafwd*(self.xcg-self.xacfwd)*self.Sfwd/(self.S*self.c) + \
               CDarear*(1-de_da)*(self.hfus-self.zcg)*self.Srear/(self.S*self.c)*self.eta_rear-\
               self.CLarear*(1-de_da)*(self.xacrear-self.xcg)*self.Srear/(self.S*self.c)*self.eta_rear
        return CX_a,CZ_a,Cm_a

    def q_derivatives(self):
        """
        Analytical estimates of stability derivatives wrt pitch rate q (q*c/V).
        :return: C_X_q, C_Z_q, C_m_q
        """
        CX_q  =0
        CZ_q = self.CLafwd*(self.xcg-self.xacfwd)*self.Sfwd/(self.S*self.c)-\
               self.CLarear*(self.xacrear-self.xcg)*self.Srear/(self.S*self.c)*self.eta_rear
        Cm_q = -(self.CLafwd*(self.xcg-self.xacfwd)**2*self.Sfwd/(self.S*self.c**2)+
                 self.CLarear*(self.xacrear-self.xcg)**2*self.Srear/(self.S*self.c**2)*self.eta_rear)
        return CX_q,-CZ_q,Cm_q

    def alpha_dot_derivatives(self):
        """
        Analytical estimates of stability derivatives wrt AoA_dot (alpha_dot*c/V).
        :return: C_X_alpha_dot, C_Z_alpha_dot, C_m_alpha_dot
        """
        CX_adot = 0
        darear_dadot = self.deps_da(self.Sweepc4,self.hfus-self.dy,self.CLafwd)*self.lh_arm()/self.c
        # print("dalpha_dalpha_dot = %.5f"%(darear_dadot))
        CLadot = self.Srear/self.S*self.CLarear*darear_dadot*self.eta_rear
        CZ_adot = -CLadot
        Cm_adot = (self.xacrear-self.xcg)/self.c*CZ_adot
        return CX_adot,CZ_adot,Cm_adot
    def r_derivatives(self):
        """
        Analytical estimates of stability derivatives wrt yaw rate r (rb/(2V)).
        :return: C_Y_r, C_l_r, C_n_r
        """
        CY_r = 2*self.C_L_a(self.ARv,self.Sweep(self.ARv*2,0.4,self.sweepTE,50,100))*(self.lv)*\
                    self.Sv/(self.S*self.b)*self.eta_v
        Pos_MAC_v = self.bv/6*((1+2*self.taper_v)/(1+self.taper_v))*2
        zv = self.hfus+Pos_MAC_v-self.zcg
        Cl_r_v = zv/self.b*CY_r
        Cl_r_fwd = self.CLfwd0/4
        Cl_r_rear = self.CLrear0/4
        Cl_r = Cl_r_v + Cl_r_fwd*self.Sfwd*self.bfwd/(self.S*self.b) + \
               Cl_r_rear*self.Srear*self.brear/(self.S*self.b)*self.eta_rear
        Cn_r_v = -(self.lv)/self.b*CY_r
        CDfwd0 = self.CD_0 + self.CLfwd0**2/(np.pi*self.Afwd*self.efwd)
        CDrear0 = self.CD_0 + self.CLrear0**2/(np.pi*self.Arear*self.erear)
        Cn_r_wing = -1/4*(CDfwd0*self.Sfwd*self.bfwd/(self.S*self.b) + CDrear0*self.Srear*self.brear/(self.S*self.b)*self.eta_rear)
        Cn_r = Cn_r_wing+Cn_r_v
        return CY_r,Cl_r,Cn_r

    def p_derivatives(self):
        """
        Analytical estimates of stability derivatives wrt roll rate p (pb/(2V)).
        :return: C_Y_p, C_l_p, C_n_p
        """
        Ctrl = Control_surface(self.V0,self.Vstall,self.CLfwd0,self.CLrear0,self.CLafwd,self.CLarear,
                               self.Clafwd,self.Clarear,self.Cd0fwd,self.Cd0rear,self.Sfwd,self.Srear,
                               self.Afwd,self.Arear,self.cfwd,self.crear,self.bfwd,self.brear,self.taper,self.eta_rear)
        Cl_p = Ctrl.Clp()
        eta_v = self.eta_v
        CY_p = -8/(3*np.pi)*eta_v*(self.bv*self.Sv/(self.S*self.b))*self.C_L_a(self.ARv,self.Sweep(self.ARv*2,0.4,self.sweepTE,50,100))
        Cn_p_v = -(self.lv)/(self.b)*CY_p
        c_r_fwd = self.cfwd * 3 / 2 * (1 + self.taper) / (1 + self.taper + self.taper ** 2)
        c_r_rear = self.crear * 3 / 2 * (1 + self.taper) / (1 + self.taper + self.taper ** 2)
        Cnp_fwd = -(self.Cl0fwd) * c_r_fwd * self.bfwd / (24 * self.Sfwd) * (1 + 3 * self.taper)
        Cnp_rear = -(self.Cl0rear) * c_r_rear * self.brear / (24 * self.Srear) * (1 + 3 * self.taper)*self.eta_rear
        Cn_p_wings = -1/8*(self.CLfwd0*self.Sfwd*self.bfwd/(self.S*self.b) + self.CLrear0*self.Srear*self.brear//(self.S*self.b))
        Cn_p_wings_analytical = Cnp_fwd*self.Sfwd*self.bfwd/(self.S*self.b) + Cnp_rear*self.Srear*self.brear/(self.S*self.b)
        # print("Analytical estimation of C_l_p:",Cl_p)
        # print("Approximation of C_l_p = ", -1/8*(self.CLarear*self.brear*self.Srear+self.CLafwd*self.bfwd*self.Sfwd)/(self.S*self.b))
        # print("Approximation: ", Cn_p_wings)
        # print("Analytical solution:",Cn_p_wings_analytical )
        Cn_p = Cn_p_v+Cn_p_wings
        return CY_p, Cl_p,Cn_p

    def beta_derivatives(self, g1, g2):
        """
        Analytical estimates of stability derivatives wrt side slip angle beta.
        :return: C_Y_beta, C_l_beta, C_n_beta
        """
        dsigma_dbeta = 0
        CY_b = -self.C_L_a(self.ARv,self.Sweep(self.ARv*2,self.taper_v,self.sweepTE,50,100))*(1-dsigma_dbeta)*self.eta_v*self.Sv/(self.S)
        Cn_b_v = -CY_b*(self.lv)/self.b
        a = self.lfus/2
        b = self.wfus/2
        V = 2*np.pi/4*b**2*(self.lfus/2-(self.lfus/2)**3/(3*a**2))
        bmax = max(self.bfwd,self.brear)

        #### Estimate of Cn_beta #####
        Cnb_fus = -2*V/(self.S*bmax)
        Cnb_w_fwd = self.CLfwd0**2*(1/(4*np.pi*self.Afwd)-
                                   (np.tan(self.Sweepc4)/(np.pi*self.Afwd+4*np.cos(self.Sweepc4)))*
                                   (np.cos(self.Sweepc4)-self.Afwd/2-self.Afwd**2/(8*np.cos(self.Sweepc4))-
                                    6*(self.xacfwd-self.xcg)*np.sin(self.Sweepc4)/(self.Afwd*self.c)))
        Cnb_w_rear = self.CLrear0**2*(1/(4*np.pi*self.Arear)-
                                   (np.tan(self.Sweepc4_rear)/(np.pi*self.Arear+4*np.cos(self.Sweepc4_rear)))*
                                   (np.cos(self.Sweepc4_rear)-self.Afwd/2-self.Arear**2/(8*np.cos(self.Sweepc4_rear))-
                                    6*(self.xacrear-self.xcg)*np.sin(self.Sweepc4_rear)/(self.Arear*self.c)))
        Cn_b_wings = Cnb_w_fwd*self.Sfwd*self.bfwd/(self.S*bmax)+Cnb_w_rear*self.Srear*self.brear/(self.S*bmax)*self.eta_rear
        Cn_b = Cnb_fus+Cn_b_wings+Cn_b_v

        #### Estimate of Cl_beta #####
        Pos_MAC_v = self.bv / 6 * ((1 + 2 * self.taper_v) / (1 + self.taper_v)) * 2
        Cl_b_v = (self.hfus+Pos_MAC_v-self.zcg)/self.b*CY_b
        Cl_b_wf_fwd = -1.2*np.sqrt(self.Afwd)*(self.dy-self.hfus/2)/self.bfwd**2*(self.lfus+self.wfus)
        Cl_b_wf_rear = -1.2*np.sqrt(self.Arear)*(self.hfus/2)/self.brear**2*(self.lfus+self.wfus)*self.eta_rear
        # Cl_b = -0.110
        Cl_b_fwd = -self.CLafwd*g1/4*(2/3*(1+2*self.taper)/(1+self.taper))
        Cl_b_rear = -self.CLarear * g2 / 4 * (2 / 3 * (1 + 2 * self.taper) / (1 + self.taper))*self.eta_rear
        # print(self.CLafwd,self.Gamma_fwd,self.taper)
        Cl_b = (Cl_b_wf_rear*self.Srear*self.brear + Cl_b_wf_fwd*self.Sfwd*self.bfwd +
                Cl_b_fwd*self.Sfwd*self.bfwd+Cl_b_rear*self.Srear*self.brear)/(self.S*self.b)+Cl_b_v
        # print(Cl_b_v, Cl_b_wf_fwd, Cl_b_wf_rear,Cl_b_fwd)
        # sin_Gamma = (Cl_b-Cl_b_v-Cl_b_wf_fwd*self.Sfwd/self.S-Cl_b_wf_rear*self.Srear/self.S)/(-2/(3*np.pi)*self.CLafwd)
        # Gamma_raymer = -(Cl_b-Cl_b_v-Cl_b_wf_fwd*self.Sfwd/self.S-Cl_b_wf_rear*self.Srear/self.S)*4/self.CLafwd*(3*(1+self.taper)/(2*(1+2*self.taper)))
        return CY_b, Cl_b,Cn_b

    def tau_e(self,Se_S):
        """
        Inputs:
        :param Se_S: Elevator surface to wing ratio [-]
        :return: Elevator Effectiveness [-]
        """
        x = Se_S
        tau_e = -6.624*x**4+12.07*x**3-8.292*x**2+3.295*x+0.004942
        return tau_e

    def de_derivatives(self,Se_S,be_b):
        """
        Analytical estimate of control derivatives wrt elevator deflection (delta_e)
        :param Se_S: Elevator surface to wing ratio [-]
        :param be_b: Elevator span to wing ratio [-]
        :return: C_X_de, C_Z_de, C_m_de
        """
        CX_de = 0
        CL_de_fwd = self.CLafwd*self.tau_e(Se_S)*be_b*self.Sfwd/self.S/100
        CL_de_rear = self.CLarear*self.tau_e(Se_S)*be_b*self.Srear/self.S*self.eta_rear/100
        CL_de = -self.CLafwd*self.tau_e(Se_S)*be_b/100*self.Sfwd/self.S +\
                self.CLarear*self.tau_e(Se_S)*be_b/100*self.Srear/self.S*self.eta_rear
        CZ_de = -CL_de
        Cm_de = -(CL_de_fwd*(self.xcg-self.xacfwd)/self.c+CL_de_rear*(self.xacrear-self.xcg)/self.c)

        return CX_de, CZ_de, Cm_de

    def da_derivatives(self,Sa_S,b1,b2):
        """
        Analytical estimates control derivatives wrt aileron deflection (delta_a)
        :param Sa_S: Aileron surface to wing ratio [-]
        :param b1: Inner position [% b/2]
        :param b2: Outer position [% b/2]
        :return: C_Y_da, C_l_da, C_n_da
        """
        CY_da = 0
        aileron = Control_surface(self.V0, self.Vstall,self.CLfwd0,self.CLrear0,self.CLafwd,self.CLarear,self.Clafwd,
                                  self.Clarear,self.Cd0fwd,self.Cd0rear,
                                  self.Sfwd,self.Srear,self.Afwd,self.Arear,self.cfwd,self.crear,self.bfwd,self.brear,
                                  self.taper,self.eta_rear)
        Cl_da = aileron.Clda(Sa_S,b1,b2,rear=True)
        Cn_da = -0.2*self.CL0*Cl_da
        return CY_da, Cl_da, Cn_da

    def tau_r(self,Cr_Cv):
        """
        Inputs:
        :param Cr_Cv: MAC ratio of rudder to vertical tail
        :return: rudder effectiveness [-]
        """
        return 1.129*(Cr_Cv)**0.4044 - 0.1772

    def dr_derivatives(self,cr_cv,br_bv):
        """
        Analytical estimates of control derivatives wrt rudder deflection (delta_r)
        :param cr_cv:  MAC ratio of rudder to vertical tail
        :param br_bv:  Span ratio of rudder to vertical tail
        :return: C_Y_dr, C_l_dr, C_n_dr
        """
        CY_dr = self.C_L_a(self.ARv,self.Sweep(self.ARv*2,self.taper_v,self.sweepTE,50,100))*\
                self.Sv/self.bv*self.eta_v*self.tau_r(cr_cv)*br_bv
        Cn_dr = -CY_dr*self.lv/self.b
        Pos_MAC_v = self.bv / 6 * ((1 + 2 * self.taper_v) / (1 + self.taper_v)) * 2
        Cl_dr = CY_dr*(self.hfus+Pos_MAC_v-self.zcg)/self.b
        return CY_dr, Cl_dr, Cn_dr

    def g(self,h,Re = 6371*10**3):
        return 9.80665*(Re/(Re+h))**2

    def get_muc(self):
        """
        Computes dimensionless mass for symmetric motion mu_c
        :return: mu_c
        """
        rho = self.rho
        return self.W/(rho*self.g(self.h)*self.S*self.c)

    def get_mub(self):
        """
        Computes dimensionless mass for asymmetric motion mu_b
        :return: mu_b
        """
        rho = self.rho
        return self.W/(rho*self.g(self.h)*self.S*self.b)

    def asym_stability_req(self, Ixx, Izz, Ixz, Sa_S,b1,b2,cr_cv,br_bv,g1,g2):
        """
        Plots lateral stability graph for Dutch Roll and spiral.
        :param Ixx: Mass Moment of Inertia around xx-axis
        :param Izz: Mass Moment of Inertia around zz-axis
        :param Ixz: Mass Moment of Inertia around xz-axis
        :param Sa_S: Aileron-wing surface ratio [-]
        :param b1: Inner span position [m]
        :param b2: Outer span position [m]
        :param cr_cv: Rudder-VT chord ratio [-]
        :param br_bv: Rudder-VT span ratio [-]
        :param g1: Dihedral angle of wing 1
        :param g2: Dihedral angle of wing 2
        :return: Plots lateral stability graph
        """
        mu_b = self.get_mub()
        Kxx2 = Ixx/(self.W/9.80665*self.b**2)
        Kzz2 = Izz/(self.W/9.80665*self.b**2)
        Kxz = Ixz/(self.W/9.80665*self.b**2)
        print("Inside Ixx, Izz =", Ixx, Izz)
        print("Inside Kxx2, Kzz2, Kxz = %.5f, %.5f, %.5f"%(Kxx2,Kzz2,Kxz))

        C_L = self.CL0
        C_Y = [self.beta_derivatives(0,0)[0],0,self.p_derivatives()[0],self.r_derivatives()[0],\
              self.da_derivatives(Sa_S,b1,b2)[0],self.dr_derivatives(cr_cv,br_bv)[0]]
        C_l = [self.beta_derivatives(0,0)[1],self.p_derivatives()[1],self.r_derivatives()[1],\
              self.da_derivatives(Sa_S,b1,b2)[1],self.dr_derivatives(cr_cv,br_bv)[1]]
        C_l_b_1 = self.beta_derivatives(g1,0)[1]
        C_l_b_2 = self.beta_derivatives(0,g2)[1]
        C_l_b_opt = self.beta_derivatives(g1,g2)[1]
        C_n = [self.beta_derivatives(0,0)[2],0, self.p_derivatives()[2], self.r_derivatives()[2], \
               self.da_derivatives(Sa_S, b1, b2)[2], self.dr_derivatives(cr_cv, br_bv)[2]]
        C_y_b, C_y_b_dot, C_y_p, C_y_r, C_y_da, C_y_dr = C_Y
        C_l_b, C_l_p, C_l_r, C_l_da, C_l_dr = C_l
        C_n_b, C_n_b_dot, C_n_p, C_n_r, C_n_da, C_n_dr = C_n
        A = 16 * (mu_b ** 3) * (Kxx2 * Kzz2 - Kxz ** 2)

        B = -4 * (mu_b ** 2) * (
                    2 * C_y_b * (Kxx2 * Kzz2 - Kxz ** 2) + C_n_r * Kxx2 + C_l_p * Kzz2 + (C_l_r + C_n_p) * Kxz)

        C = 2 * mu_b * ((C_y_b * C_n_r - C_y_r * C_n_b) * Kxx2 + (C_y_b * C_l_p - C_l_b * C_y_p) * Kzz2 +
                        ((C_y_b * C_n_p - C_n_b * C_y_p) + (
                                    C_y_b * C_l_r - C_l_b * C_y_r)) * Kxz + 4 * mu_b * C_n_b * Kxx2 + 4 * mu_b * C_l_b * Kxz +
                        0.5 * (C_l_p * C_n_r - C_n_p * C_l_r))

        E = C_L * (C_l_b * C_n_r - C_n_b * C_l_r)

        D = -4 * mu_b * C_L * (C_l_b * Kzz2 + C_n_b * Kxz) + 2 * mu_b * (
                    C_l_b * C_n_p - C_n_b * C_l_p) + 0.5 * C_y_b * (C_l_r * C_n_p - C_n_r * C_l_p) + \
            0.5 * C_y_p * (C_l_b * C_n_r - C_n_b * C_l_r) + 0.5 * C_y_r * (C_l_p * C_n_b - C_n_p * C_l_b)

        R = B * C * D - A * D ** 2 - B ** 2 * E
        # if E > 0 and R > 0:
        #     print("We have Spiral stability with ", "E = ", E)
        #     print("AND Dutch Roll stability with ", "R = ", R)
        # if R > 0 and E < 0:
        #     print("We have Dutch Roll stability with ", "R = ", R)
        # if E > 0 and R < 0:
        #     print("We have Spiral stability with ", "E = ", E)
        # if E < 0:
        #     print("We have Spiral instability with ", "E = ", E)
        # if R < 0:
        #     print("We have Dutch Roll instability.")
        # print("Routh's Discriminant: %.4f "%(R))
        # clb = np.linspace(-0.5, 0.1, 50)

        def cnb_E(clb):
            return -C_n_r / C_l_r * (-clb)

        def C_ER(clb, cnb):
            return 2 * mu_b * ((C_y_b * C_n_r - C_y_r * cnb) * Kxx2 + (C_y_b * C_l_p - clb * C_y_p) * Kzz2 +
                               ((C_y_b * C_n_p - cnb * C_y_p) + (
                                           C_y_b * C_l_r - clb * C_y_r)) * Kxz + 4 * mu_b * cnb * Kxx2 + 4 * mu_b * clb * Kxz +
                               0.5 * (C_l_p * C_n_r - C_n_p * C_l_r))

        def D_ER(clb, cnb):
            return -4 * mu_b * C_L * (clb * Kzz2 + cnb * Kxz) + 2 * mu_b * (clb * C_n_p - cnb * C_l_p) + 0.5 * C_y_b * (
                        C_l_r * C_n_p - C_n_r * C_l_p) + \
                   0.5 * C_y_p * (clb * C_n_r - cnb * C_l_r) + 0.5 * C_y_r * (C_l_p * cnb - C_n_p * clb)

        def E_ER(clb, cnb):
            return C_L * (clb * C_n_r - cnb * C_l_r)

        def R_ER(clb, cnb):
            return B * C_ER(clb, cnb) * D_ER(clb, cnb) - A * (D_ER(clb, cnb)) ** 2 - B ** 2 * E_ER(clb, cnb)

        clb2 = np.linspace(0,0.15, 150)
        cnb2 = np.linspace(0, 0.1, 150)

        # from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        # import matplotlib.image as mpimg
        # arr_lena = mpimg.imread('Ce550.png')
        # imagebox = OffsetImage(arr_lena, zoom=0.09)
        # ab = AnnotationBbox(imagebox, (-C_l_b, C_n_b))

        X, Y = np.meshgrid(clb2, cnb2)
        Z = R_ER(-X, Y)
        fig, ax = plt.subplots(1, 1)
        # ax.add_artist(ab)
        # levels = [-5000,-2500,-1500, -500, 0, 250, 500, 750, 1000, 1250,2000]
        cp = ax.contourf(X, Y, Z, cmap='coolwarm',levels=25)
        cbar = plt.colorbar(cp, orientation="horizontal")
        RR = ax.contour(X, Y, Z, [0], colors=["r"])
        # print("Sv_stability = ",self.VT_stability(lv))
        plt.clabel(RR, fmt=r"Dutch Roll")
        cbar.set_label(r"Routh's discriminant")
        plt.plot(clb2, cnb_E(-clb2), color="k", label="Limit for Spiral stability when E = 0")
        plt.xlabel(r"-$C_{l_{\beta}}$ [rad$^{-1}$]", fontsize=12)
        plt.ylabel(r"$C_{n_{\beta}}$ [rad$^{-1}$]", fontsize=12)
        plt.scatter(-C_l_b, C_n_b, color="k", marker="x", label="Wigeon with no dihedral")
        plt.scatter(-C_l_b_1, C_n_b, color="r", marker="o", label=r"Wigeon with $\Gamma_{fwd} = %.1f$ deg "%(np.rad2deg(g1)))
        plt.scatter(-C_l_b_2, C_n_b, color="b", marker="X", label=r"Wigeon with $\Gamma_{rear} = %.1f$ deg"%(np.rad2deg(g2)))
        plt.scatter(-C_l_b_opt, C_n_b, color="g", marker="D", label=r"Wigeon with both $\Gamma_{fwd} = %.1f$ deg and $\Gamma_{rear} = %.1f$ deg"%(np.rad2deg(g1),np.rad2deg(g2)))
        plt.legend(loc=1)
        plt.show()
        return
    def initial_coeff(self):
        C_X0 = self.W/(0.5*self.rho*self.V0**2*self.S)*np.sin(self.th0+self.alpha0)
        C_Z0 = -self.W/(0.5*self.rho*self.V0**2*self.S)*np.cos(self.th0+self.alpha0)
        C_m0 =0
        return C_X0,C_Z0
    def return_stab_derivatives(self,Se_S,be_b,Sa_S,b1,b2,cr_cv,br_bv,Ixx, Iyy, Izz, Ixz,g1,g2,matlab=True):
        a = self.alpha_derivatives()
        a_dot = self.alpha_dot_derivatives()
        u = self.u_derivatives()
        initial = self.initial_coeff()
        q = self.q_derivatives()
        de = self.de_derivatives(Se_S,be_b)
        b = self.beta_derivatives(g1,g2)
        p = self.p_derivatives()
        r = self.r_derivatives()
        da = self.da_derivatives(Sa_S,b1,b2)
        dr = self.dr_derivatives(cr_cv,br_bv)
        C_x_a, C_x_a_dot, C_x_u, C_x_0, C_x_q, C_x_d = a[0],a_dot[0],u[0], initial[0],q[0],de[0]
        C_z_a, C_z_a_dot, C_z_u, C_z_0, C_z_q, C_z_d = a[1],a_dot[1],u[1], initial[1],q[1],de[1]
        C_m_a, C_m_a_dot, C_m_u, C_m_q, C_m_d = a[2],a_dot[2],u[2],q[2],de[2]
        C_X = [C_x_a, C_x_a_dot, C_x_u, C_x_0, C_x_q, C_x_d]
        C_Z = [C_z_a, C_z_a_dot, C_z_u, C_z_0, C_z_q, C_z_d]
        C_m = [C_m_a, C_m_a_dot, C_m_u, C_m_q, C_m_d]
        # print(C_X, C_Z, C_m)
        C_y_b, C_y_b_dot, C_y_p, C_y_r, C_y_da, C_y_dr = b[0], 0, p[0],r[0],da[0],dr[0]
        C_l_b, C_l_p, C_l_r, C_l_da, C_l_dr = b[1],p[1],r[1],da[1],dr[1]
        C_n_b, C_n_b_dot, C_n_p, C_n_r, C_n_da, C_n_dr = b[2],0,p[2],r[2],da[2],dr[2]
        C_Y = [C_y_b, C_y_b_dot, C_y_p, C_y_r, C_y_da, C_y_dr]
        C_l = [C_l_b, C_l_p, C_l_r, C_l_da, C_l_dr]
        C_n = [C_n_b, C_n_b_dot, C_n_p, C_n_r, C_n_da, C_n_dr]
        if matlab == True:
            print("aeroCoeff = struct;\n"
                  "aeroCoeff.Cl0 = 0;\n"
                  "aeroCoeff.Cl_beta = %.09f;\n"
                  "aeroCoeff.Cl_da = %.10f;\n"
                  "aeroCoeff.Cl_dr = %.10f;\n"
                  "aeroCoeff.Cl_p = %.10f;\n"
                  "aeroCoeff.Cl_r = %.10f;\n"
                  "aeroCoeff.Cm0 = 0;\n"
                  "aeroCoeff.Cm_alpha = %.10f;\n"
                  "aeroCoeff.Cm_de = %.10f;\n"
                  "aeroCoeff.Cm_q = %.10f;\n"
                  "aeroCoeff.Cn0 = 0;\n"
                  "aeroCoeff.Cn_beta = %.10f;\n"
                  "aeroCoeff.Cn_da = %.10f;\n"
                  "aeroCoeff.Cn_dr = %.10f;\n"
                  "aeroCoeff.Cn_p = %.10f;\n"
                  "aeroCoeff.Cn_r = %.10f;\n"
                  "aeroCoeff.Cx0 = 0;\n"
                  "aeroCoeff.Cx_alpha = %.10f;\n"
                  "aeroCoeff.Cx_q = 0;\n"
                  "aeroCoeff.Cx_de = 0;\n"
                  "aeroCoeff.Cy0 = 0;\n"
                  "aeroCoeff.Cy_beta = %.10f;\n"
                  "aeroCoeff.Cy_da = 0;\n"
                  "aeroCoeff.Cy_dr = %.10f;\n"
                  "aeroCoeff.Cy_p = %.10f;\n"
                  "aeroCoeff.Cy_r = %.10f;\n"
                  "aeroCoeff.Cz0 = %.10f;\n"
                  "aeroCoeff.Cz_alpha = %.10f;\n"
                  "aeroCoeff.Cz_de = %.10f;\n"
                  "aeroCoeff.Cz_q = %.10f;\n"
                  "aeroCoeff.Cx_u = %.10f;\n"
                  "aeroCoeff.Cz_u = %.10f;\n"
                  "aeroCoeff.Cm_u = %.10f;\n"
                  "aeroCoeff.Cx_adot = %.10f;\n"
                  "aeroCoeff.Cz_adot = %.10f;\n"
                  "aeroCoeff.Cm_adot =%.10f;\n"
                  "Sref =%.10f;\n"
                  "bref =%.10f;\n"
                  "cref =%.10f;\n"
                  "Ixx =%.10f;\n"
                  "Iyy =%.10f;\n"
                  "Izz =%.10f;\n"
                  "Ixz =%.10f;\n"
                  "vehicle.acMass = %.10f;\n"
                  "aeroCoeff.V0 = %.10f;"%(C_l_b, C_l_da,C_l_dr,C_l_p,C_l_r,C_m_a,C_m_d, C_m_q,C_n_b, C_n_da,
                                               C_n_dr,C_n_p,C_n_r,C_x_a,C_y_b,C_y_dr,C_y_p,C_y_r, C_z_0, C_z_a,C_z_d,
                                               C_z_q,C_x_u,C_z_u,C_m_u,C_x_a_dot,C_z_a_dot,C_m_a_dot,self.S,self.b,
                                           self.c,Ixx, Iyy, Izz, Ixz,self.W/9.80665,self.V0))
        return C_X, C_Z,C_m, C_Y, C_l, C_n
















