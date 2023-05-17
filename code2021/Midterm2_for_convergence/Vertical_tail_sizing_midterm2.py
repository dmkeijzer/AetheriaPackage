import numpy as np
from scipy.linalg import null_space
from matplotlib import pyplot as plt
from matplotlib import colors as mc
from Aero_tools import ISA
class VT_sizing:
    def __init__(self,W,h,xcg,lfus,hfus,wfus,V0,Vstall,CD0,CLfwd,CLrear,
                 CLafwd,CLarear,
                 Sfwd,Srear,Afwd,Arear,Lambda_c4_fwd,Lambda_c4_rear,cfwd,crear,bfwd,brear,taper,ARv):
        self.W = W         # Weight [N]
        self.h = h     # Height [m]
        Aero = ISA(self.h)
        self.rho = Aero.density()
        self.T = Aero.temperature()
        self.mu = Aero.viscosity_dyn()
        self.lfus = lfus # Length of the fuselage
        self.hsus = hfus # Height of the fuselage [m]
        self.wfus = wfus # Maximum width of the fuselage [m]
        self.Srear = Srear # Rear wing area [m^2]
        self.Sfwd = Sfwd   # Forward wing area [m^2]
        self.S = Srear+Sfwd # Aircraft wing area [m^2]
        self.cfwd = cfwd         # Average chord [m]
        self.crear = crear  # Average chord [m]
        self.bfwd = bfwd         # Wing span [m]
        self.brear = brear # Wing span [m]
        self.taper = taper # Wing taper ratio [-]
        self.CLfwd,self.CLrear  = CLfwd,CLrear # DESIGN FOR CRUISE Lift coefficients [-]
        self.Afwd, self.Arear = Afwd, Arear # Aspect ratio of both wings [-]
        self.Sweepc4fwd = Lambda_c4_fwd # Sweep at c/2 [rad]
        self.Sweepc4rear = Lambda_c4_rear # Sweep at c/2 [rad]
        self.Sweepc2fwd = self.Sweep(Afwd,self.Sweepc4fwd,50,25)
        self.Sweepc2rear = self.Sweep(Arear, self.Sweepc4rear, 50, 25)
        self.V0 = V0       # Initial speed [m/s]
        self.M0 = self.V0/(1.4*287*self.T) # Initial mach number [-]
        self.Re = self.rho*self.V0*self.lfus/self.mu
        self.CLafwd, self.CLarear = CLafwd, CLarear # Wing lift curve slopes for both wings [1/rad]
        # self.Cmacfwd, self.Cmacrear = Cmacfwd,Cmacrear
        self.CD0 = CD0 # C_D_0 of forward wing
        self.xacfwd = 0.25*self.cfwd
        self.xacrear = self.lfus - (1 - 0.25) * self.crear
        # self.de_da = self.deps_da(self.Sweepc4fwd,self.bfwd,self.lh(),self.hfus,self.Afwd,self.CLafwd)
        self.taper_v = 0.4
        self.Vs = Vstall # Stall speed [m/s]
        self.Vmc = 1.2*self.Vs # Minimum controllable speed [m/s]
        self.xcg = xcg
        self.c = self.Sfwd/self.S*self.cfwd+self.Srear/self.S*self.crear
        self.ARv = ARv
        self.Sweep_v_c2 = self.Sweep(self.ARv, 0, 50, 100)

    def Sweep(self,AR,Sweepm,n,m):
        """
        Inputs
        :param AR: Aspect Ratio of VT
        :param Sweepm: Sweep at mth chord [rad]
        :param n: (example quarter chord: n =25)
        :param m: mth chord (example half chord: m=50)
        :return: Sweep at nth chord [rad]
        """
        tanSweep_m = np.tan(Sweepm)
        tanSweep_n = tanSweep_m -4/(AR*4)*(n-m)/100*(1-self.taper)/(1+self.taper)
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
    def initial_VT(self,lv,VT = 0.04):
        """
        Inputs:
        AIRFOIL USED: 0012
        :param lv:
        :param VT:
        :return:
        """
        Sv = max(self.bfwd,self.brear)*self.S*VT/lv
        ARv = self.ARv
        bv = np.sqrt(ARv*Sv)
        C_v = Sv/bv
        C_vr = 3/2*C_v*(1+self.taper_v)/(1+self.taper_v+self.taper_v**2)
        C_vt = self.taper_v*C_vr
        Sweep_v_c2 = self.Sweep(self.ARv,0,50,100)
        return Sv,ARv,bv,C_v,Sweep_v_c2,C_vr,C_vt

    def tau(self,Cr_Cv):
        """
        Inputs:
        :param Cr: MAC of rudder [m]
        :param Cv: MAC vertical tail [m]
        :return: rudder effectiveness [-]
        """
        return 1.129*(Cr_Cv)**0.4044 - 0.1772

    def VT_controllability(self,nE,Tt0,yE,lv,br_bv,cr_cv):
        """
        Inputs:
        :param nE: Number of engines [-]
        :param Tt0: Take-off thrust [N]
        :param yE: Largest moment arm [m]
        :return: Required vertical tail area [m^2] for controllability
        """
        N_E = Tt0/nE*yE # Asymmetric yaw moment [Nm]
        N_D = 0.25*N_E # component due to drag [Nm]
        N_total = N_E + N_D
        Sr_Sv = 0.2
        dr_max = 25*np.pi/180
        C_rudder = self.initial_VT(lv)[3]*cr_cv
        tau_r = self.tau(cr_cv)
        CLa_v = self.C_L_a(self.ARv, self.initial_VT(lv)[4])
        Vv_V = 1
        Sv = N_total/(0.5*self.rho*self.Vmc**2*CLa_v*lv*Vv_V**2*tau_r*br_bv*dr_max)
        return Sv

    def VT_stability(self,lv):
        """
        Inputs
        :param lv: CG moment arm
        :return: Sv for stability [m^2]
        """
        # kn = 0.01*(0.27*self.xcg/self.lfus-0.168*np.log(self.lfus/self.wfus)+0.416)-0.0005
        # kR = 0.46*np.log10(self.Re/10**6)+1
        # Cnb_fus = -360/(2*np.pi)*kn*kR*self.lfus**2*self.wfus/(self.S*max(self.brear,self.bfwd))
        a = self.lfus/2
        b = self.wfus/2
        V = 2*np.pi/4*b**2*(self.lfus/2-(self.lfus/2)**3/(3*a**2))
        bmax = max(self.bfwd,self.brear)
        Cnb_fus = -2*V/(self.S*max(self.bfwd,self.brear))
        Cnb_w_fwd = self.CLfwd**2*(1/(4*np.pi*self.Afwd)-
                                   (np.tan(self.Sweepc4fwd)/(np.pi*self.Afwd+4*np.cos(self.Sweepc4fwd)))*
                                   (np.cos(self.Sweepc4fwd)-self.Afwd/2-self.Afwd**2/(8*np.cos(self.Sweepc4fwd))-
                                    6*(self.xacfwd-self.xcg)*np.sin(self.Sweepc4fwd)/(self.Afwd*self.c)))
        Cnb_w_rear = self.CLrear**2*(1/(4*np.pi*self.Arear)-
                                   (np.tan(self.Sweepc4rear)/(np.pi*self.Arear+4*np.cos(self.Sweepc4rear)))*
                                   (np.cos(self.Sweepc4rear)-self.Afwd/2-self.Arear**2/(8*np.cos(self.Sweepc4rear))-
                                    6*(self.xacrear-self.xcg)*np.sin(self.Sweepc4rear)/(self.Arear*self.c)))
        # print("Cnbw_fwd, Cnbw_rear = ",Cnb_w_fwd,Cnb_w_rear)
        # print("Cn_fus = %.4f [1/rad]"%(Cnb_fus))
        CYb_v = -self.C_L_a(self.ARv,self.initial_VT(lv)[4])
        # print("CYb_v = %.3f "%(CYb_v))
        Cnb = 0.06
        Sv = self.S*(Cnb-Cnb_fus-Cnb_w_fwd*self.Sfwd*self.bfwd/(self.S*bmax)-
                     Cnb_w_rear*self.Srear*self.brear/(self.S*bmax))/(-CYb_v)*bmax/lv
        return Sv

    def final_VT_rudder(self,nE,Tt0,yE,lv,br_bv, cr_cv):
        """
        Inputs:
        :param nE: Number of propellers
        :param Tt0: Thrust [N]
        :param yE: Moment arm [m]
        :param lv: CG moment arm [m]
        :return: Final design
        """
        if isinstance(br_bv,float) and isinstance(self.ARv,float):
            Sv = max(self.VT_controllability(nE,Tt0,yE,lv,br_bv,cr_cv),self.VT_stability(lv))
            # print("Stability: ", self.VT_stability(lv))
            # print("Controllability: ", self.VT_controllability(nE,Tt0,yE,lv,br_bv,cr_cv))
        else:
            Sv = self.VT_controllability(nE,Tt0,yE,lv,br_bv,cr_cv)
        ARv = self.ARv
        bv = np.sqrt(ARv*Sv)
        C_v = Sv/bv
        C_vr = 3/2*C_v*(1+self.taper_v)/(1+self.taper_v+self.taper_v**2)
        C_vt = self.taper_v*C_vr
        Sweep_v_c2 = self.Sweep(ARv,0,50,100) # Design variable TE sweep 0.
        c_r = cr_cv*C_v
        c_r_root = 3/2*c_r*(1+self.taper_v)/(1+self.taper_v+self.taper_v**2)
        c_r_tip = self.taper_v*c_r_root
        b_r = br_bv*bv
        return Sv,C_vr,C_vt,bv,Sweep_v_c2,c_r,c_r_root,c_r_tip,b_r,ARv

    def plotting(self,nE,Tt0,yE,lv,br_bv,cr_cv):
        if isinstance(br_bv,float) and isinstance(self.ARv,float):
            y_LE_0 = 0
            x_LE_0 = 0
            x_TE_1 = self.final_VT_rudder(nE,Tt0,yE,lv,br_bv,cr_cv)[1]
            y_TE_1 = 0
            x_TE_2 = x_TE_1
            y_TE_2 = self.final_VT_rudder(nE,Tt0,yE,lv,br_bv,cr_cv)[3]
            y_LE_3 = y_TE_2
            x_LE_3 = x_TE_1 - x_TE_1 * 0.4
            y_up = br_bv * y_TE_2
            y_down = 0
            x1 = x_TE_1 - cr_cv * x_TE_1
            x2 = x_TE_1
            x3 = x2
            x4 = x_TE_1 - cr_cv * 0.4 * x_TE_1
            x_r = np.array([x1, x2, x3, x4, x1])
            y_r = np.array([y_down, y_down, y_up, y_up, y_down])
            x_points = np.array([x_LE_0, x_TE_1, x_TE_2, x_LE_3, 0])
            y_points = np.array([y_LE_0, y_TE_1, y_TE_2, y_LE_3, 0])
            Sv_estimate = (x_TE_1+(x_TE_1-x_LE_3))/2*y_TE_2
            plt.plot(x_points, y_points, label="Vertical tail")
            plt.plot(x_r, y_r, label="Rudder")
            plt.legend()
            plt.show()
        elif not isinstance(self.ARv,float) and isinstance(br_bv,float):
            # X, Y = np.meshgrid(cr_cv, br_bv)
            # Z = self.final_VT_rudder(nE,Tt0,yE,lv,Y,X)[0]
            # fig, ax = plt.subplots(1, 1)
            # ax.add_artist(ab)
            # levels = [0,0.1,1,1.]
            # cp = ax.contourf(X, Y, Z, cmap='coolwarm')
            # Svstab = ax.contour(X,Y,Z,[self.VT_stability(lv)],colors=["k"])
            # plt.clabel(Svstab)
            Svstab = self.VT_stability(lv)
            Svcontrol = self.VT_controllability(nE,Tt0,yE,lv,br_bv,cr_cv)
            bv = self.final_VT_rudder(nE,Tt0,yE,lv,br_bv,cr_cv)[3]
            # cbar = plt.colorbar(cp, orientation="horizontal")
            # cbar.set_label(r"$S_v$")
            # plt.ylabel(r"$b_r/b_v$ [-]", fontsize=12)
            # plt.xlabel(r"$c_r/c_v$ [-]", fontsize=12)
            # plt.show()
            Sv_estimate = None
            plt.plot(self.ARv,Svstab,label="Stability Curve")
            plt.plot(self.ARv,Svcontrol,label="Controllability for OEI condition")
            plt.xlabel(r"$AR_v [-]$")
            plt.ylabel(r"$S_v [m^2]$")
            plt.legend()
            plt.show()
            plt.plot(self.ARv,bv)
            plt.xlabel(r"$AR_v [-]$")
            plt.ylabel(r"$b_v [m]$")
            plt.show()
        else:
            X, Y = np.meshgrid(cr_cv, br_bv)
            Z = self.final_VT_rudder(nE, Tt0, yE, lv, Y, X)[0]
            fig, ax = plt.subplots(1, 1)
            Sv_estimate = None
            # ax.add_artist(ab)
            # levels = [0,0.1,1,1.]
            cp = ax.contourf(X, Y, Z, cmap='coolwarm',levels=20)
            Svstab = ax.contour(X, Y, Z, [self.VT_stability(lv)], colors=["k"])
            plt.clabel(Svstab,fmt=r"Min. :  %.3f"%(self.VT_stability(lv)))
            cbar = plt.colorbar(cp, orientation="horizontal")
            cbar.set_label(r"$S_v$ $[m^2]$")
            plt.ylabel(r"$b_r/b_v$ [-]", fontsize=12)
            plt.xlabel(r"$c_r/c_v$ [-]", fontsize=12)
            plt.show()
        return Sv_estimate




