"""
Vertical tail sizing. Class VT_sizing allows for sizing and plotting of the vertical tail.

Author: Michal Cuadrat-Grzybowski
"""
import numpy as np
from scipy.linalg import null_space
from matplotlib import pyplot as plt
from matplotlib import colors as mc
from Aero_tools import ISA
from Final_optimization import constants_final as const
class VT_sizing:
    def __init__(self,W,h,xcg,lfus,hfus,wfus,V0,Vstall,CD0,CLfwd,CLrear,
                 CLafwd,CLarear,Sfwd,Srear,Afwd,Arear,Lambda_c4_fwd,Lambda_c4_rear,cfwd,crear,bfwd,brear,taper,ARv,sweepTE):
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
        self.M0 = self.V0/(np.sqrt(1.4*287*self.T)) # Initial mach number [-]
        self.Re = self.rho*self.V0*self.lfus/self.mu
        self.CLafwd, self.CLarear = CLafwd, CLarear # Wing lift curve slopes for both wings [1/rad]
        # self.Cmacfwd, self.Cmacrear = Cmacfwd,Cmacrear
        self.CD0 = CD0 # C_D_0 of forward wing
        # self.xacfwd = 0.25*self.cfwd
        # self.xacrear = self.lfus - (1 - 0.25) * self.crear
        self.xacfwd = 0.5
        self.xacrear = 6.1
        # self.de_da = self.deps_da(self.Sweepc4fwd,self.bfwd,self.lh(),self.hfus,self.Afwd,self.CLafwd)
        self.taper_v = 0.4
        self.Vs = Vstall # Stall speed [m/s]
        self.Vmc = 1.2*self.Vs # Minimum controllable speed [m/s]
        self.xcg = xcg
        self.c = self.Sfwd/self.S*self.cfwd+self.Srear/self.S*self.crear
        self.ARv = ARv
        self.sweepTE = sweepTE
        self.b = np.sqrt(0.5 * (self.Srear / self.S * self.Arear + self.Sfwd / self.S * self.Afwd) * self.S)

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
        tanSweep_n = tanSweep_m -4/(2*AR)*(n-m)/100*(1-self.taper)/(1+self.taper)
        return np.arctan(tanSweep_n)

    def C_L_a(self,A, Lambda_half, eta=0.95):
        """
        Inputs:
        :param A: Aspect ratio [-]
        :param Lambda_half: Sweep angle at 0.5*chord
        :param eta: =0.95
        :return: Lift curve slope for tail AND wing using DATCOM method
        """
        M= self.M0
        beta = np.sqrt(1 - M ** 2)
        # print("Lambda_1/2c = ",Lambda_half)
        value = 2 * np.pi * A / (2 + np.sqrt(4 + ((A * beta / eta) ** 2) * (1 + (np.tan(Lambda_half) / beta) ** 2)))
        return value

    def initial_VT(self,VT = 0.04):
        """
        Inputs:
        AIRFOIL USED: 0012
        :param VT: VT volume coefficient [-] (optional, if None VT = 0.04)
        :return: Sv (surface area of VT), ARv (VT aspect ratio) , bv (VT height),
        C_v (mean aerodynamic chord) ,Sweep_v_c2 (sweep angle at c/2), C_vr (root of the VT), C_vt (tip of the VT)
        """
        lv = self.lfus-self.xcg
        Sv = max(self.bfwd,self.brear)*self.S*VT/lv
        ARv = self.ARv
        bv = np.sqrt(ARv*Sv)
        C_vr = 2/(1+self.taper_v)*Sv/bv
        C_v = 2/3*C_vr*(1+self.taper_v+self.taper_v**2)/(1+self.taper_v)
        C_vt = self.taper_v*C_vr
        Sweep_v_c2 = self.Sweep(self.ARv,self.sweepTE,50,100)
        # print("Initially Sv:", Sv)
        return Sv,ARv,bv,C_v,Sweep_v_c2,C_vr,C_vt

    def tau(self,Cr_Cv):
        """
        Inputs:
        :param Cr_Cv: MAC of rudder and VT ratio [-]
        :return: rudder effectiveness [-]
        """
        return 1.129*(Cr_Cv)**0.4044 - 0.1772

    def xacv(self,ARv,sweepTE):
        """
        Inputs
        :param ARv: Aspect ratio of the VT [-]
        :param sweepTE: Sweep angle of the VT at the trailing edge [rad]
        :return: x_ac_v (VT aerodynamic center position from nose) [m]
        """
        ylemac = self.initial_VT()[2]*2/6*(1+2*self.taper_v)/(1+self.taper_v)
        xlemac = ylemac*np.tan(self.Sweep(ARv,sweepTE,0,100))
        xacv = self.lfus-self.initial_VT()[5] +xlemac+0.25*0.5*(self.initial_VT()[5]+self.initial_VT()[6])
        # print(ylemac, xlemac)
        # print(xacv)
        return xacv

    def VT_controllability(self,r1,r2,nf, nE,Tt0,br_bv,cr_cv,ARv,sweepTE):
        """
        Inputs:
        :param r1: Radius of propellers on forward wing [m]
        :param r2: Radius of propellers on rear wing [m]
        :param nf: Number of failed engines [-]
        :param nE: Number of engines [-]
        :param Tt0: Take-off thrust [N]
        :param ARv: Aspect ratio [-]
        :param br_bv: Span of rudder over span of VT [-]
        :param cr_cv: Chord of rudder over chord of VT [-]
        :param ARv: Aspect ratio of VT [-]
        :param sweepTE: Sweep of VT [rad]
        :return: Required vertical tail area [m^2] for controllability
        """
        lv = self.xacv(ARv,sweepTE)-self.xcg
        N_E = 0
        # r = 0.50292
        yE1 = self.bfwd/2
        cpp = const.c_pp
        yE2 = yE1- r1-cpp-r1
        yE3 = yE2-r1-cpp-r1
        y = [yE1,yE2,yE3]
        for i in range(nf):
            yE =y[i]
            # print(N_E)
            N_E += Tt0/nE*yE# Asymmetric yaw moment [Nm]
        yE1 = self.brear / 2
        yE2 = yE1 - r2 - cpp - r2
        yE3 = yE2 - r2 - cpp - r2
        y = [yE1, yE2, yE3]
        for i in range(nf):
            yE = y[i]
            # print(N_E)
            N_E += Tt0 / nE * yE  # Asymmetric yaw moment [Nm]
        # print(N_E)
        N_D = 0.25*N_E # component due to drag [Nm]
        N_total = N_E + N_D
        # print(N_total)
        Sr_Sv = 0.2
        dr_max = 25*np.pi/180
        C_rudder = self.initial_VT()[3]*cr_cv
        tau_r = self.tau(cr_cv)
        CLa_v = self.C_L_a(ARv, self.Sweep(ARv,sweepTE,50,100))
        Vv_V = 1
        Sv = N_total/(0.5*self.rho*self.Vmc**2*CLa_v*lv*Vv_V**2*tau_r*br_bv*dr_max)
        # print("Ctrl: lv = ",lv)
        # print("Print inside the function Svctrl",Sv)
        return Sv

    def VT_stability(self,ARv, sweepTE):
        """
        Inputs
        :param ARv: Aspect ratio of Vt [-]
        :param sweepTE: Sweep of VT (trailing edge) [rad]
        :return: Sv for stability [m^2]
        """
        # kn = 0.01*(0.27*self.xcg/self.lfus-0.168*np.log(self.lfus/self.wfus)+0.416)-0.0005
        # kR = 0.46*np.log10(self.Re/10**6)+1
        # Cnb_fus = -360/(2*np.pi)*kn*kR*self.lfus**2*self.wfus/(self.S*max(self.brear,self.bfwd))
        lv = self.xacv(ARv,sweepTE) - self.xcg
        a = self.lfus/2
        b = self.wfus/2
        V = 2*np.pi/4*b**2*(self.lfus/2-(self.lfus/2)**3/(3*a**2))
        bmax = np.sqrt(0.5*(self.Srear/self.S*self.Arear+self.Sfwd/self.S*self.Afwd)*self.S)
        Cnb_fus = -2*V/(self.S*self.b)
        Cnb_w_fwd = self.CLfwd**2*(1/(4*np.pi*self.Afwd)-
                                   (np.tan(self.Sweepc4fwd)/(np.pi*self.Afwd+4*np.cos(self.Sweepc4fwd)))*
                                   (np.cos(self.Sweepc4fwd)-self.Afwd/2-self.Afwd**2/(8*np.cos(self.Sweepc4fwd))-
                                    6*(self.xacfwd-self.xcg)*np.sin(self.Sweepc4fwd)/(self.Afwd*self.c)))
        Cnb_w_rear = self.CLrear**2*(1/(4*np.pi*self.Arear)-
                                   (np.tan(self.Sweepc4rear)/(np.pi*self.Arear+4*np.cos(self.Sweepc4rear)))*
                                   (np.cos(self.Sweepc4rear)-self.Afwd/2-self.Arear**2/(8*np.cos(self.Sweepc4rear))-
                                    6*(self.xacrear-self.xcg)*np.sin(self.Sweepc4rear)/(self.Arear*self.c)))
        CYb_v = -self.C_L_a(ARv,self.Sweep(ARv,sweepTE,50,100))

        Cnb = 0.0571
        Sv = self.S*(Cnb-Cnb_fus-Cnb_w_fwd*self.Sfwd*self.bfwd/(self.S*self.b)-
                     Cnb_w_rear*self.Srear*self.brear/(self.S*self.b))/(-CYb_v)*self.b/lv
        # print("Stability: lv = ", lv)
        # print("Print inside the function Svstab", Sv)
        return Sv

    def final_VT_rudder(self,r1,r2,nf,nE,Tt0,br_bv, cr_cv,ARv,sweepTE):
        """
        Inputs:
        :param r1: Radius of propellers on forward wing
        :param r2: Radius of propellers on rear wing
        :param nf: Number of failed engines
        :param nE: Number of propellers
        :param Tt0: Thrust [N]
        :param br_bv: Span of rudder over span of VT [-]
        :param cr_cv: Chord of rudder over chord of VT [-]
        :param ARv: Aspect ratio of VT [-]
        :param sweepTE: Sweep of VT [rad]
        :return: Final design: Sv (surface area of VT), bv (VT height),
        C_v (mean aerodynamic chord),  C_vr (root of the VT), C_vt (tip of the VT), Sweep_v_c2 (sweep angle at c/2),
        c_r (MAC of rudder), c_r_root (rudder chord at root), c_r_tip (rudder chord at tip),
        b_r (rudder span) , ARv (VT aspect ratio)
        - plotting function: sensitivity analysis and/or VT with rudder
        """
        if isinstance(br_bv,(float,int)) and isinstance(self.ARv,(float,int)) and isinstance(self.sweepTE,(float,int)):
            Sv = max(self.VT_controllability(r1,r2,nf,nE,Tt0,br_bv,cr_cv,ARv,sweepTE),self.VT_stability(ARv,sweepTE))
            # print("Stability inside final VT: ", self.VT_stability(ARv,sweepTE))
            # print("Controllability inside final VT: ", self.VT_controllability(nE,Tt0,yE,br_bv,cr_cv,ARv,sweepTE))
            # print("Sv_max inside = ",Sv)

        elif isinstance(br_bv,(float,int)) and not isinstance(ARv,(float,int)) and not isinstance(sweepTE,(float,int)):
            Sv_stab = self.VT_stability(ARv,sweepTE)
            Sv_ctrl = self.VT_controllability(r1,r2,nf,nE,Tt0,br_bv,cr_cv,ARv,sweepTE)
            Sv = np.where(Sv_stab < Sv_ctrl,Sv_ctrl,Sv_stab)

        else:
            Sv_stab = self.VT_stability(ARv, sweepTE)
            Sv_ctrl = self.VT_controllability(r1,r2, nf,nE, Tt0, br_bv, cr_cv,ARv,sweepTE)
            Sv = np.where(Sv_stab < Sv_ctrl, Sv_ctrl, Sv_stab)

        bv = np.sqrt(ARv*Sv)
        C_vr = 2/(1+self.taper_v)*Sv/bv
        C_v = 2/3*C_vr*(1+self.taper_v+self.taper_v**2)/(1+self.taper_v)
        C_vt = self.taper_v*C_vr
        Sweep_v_c2 = self.Sweep(ARv,sweepTE,50,100) # Design variable TE sweep 0.
        # Sweep_v_c4 = self.Sweep(ARv,sweepTE,25,100)
        # print("c/4 sweep: ",Sweep_v_c4*180/np.pi)
        c_r = cr_cv*C_v
        c_r_root = 3/2*c_r*(1+self.taper_v)/(1+self.taper_v+self.taper_v**2)
        c_r_tip = self.taper_v*c_r_root
        b_r = br_bv*bv
        return Sv,C_vr,C_vt,bv,Sweep_v_c2,c_r,c_r_root,c_r_tip,b_r,ARv

    def plotting(self,r1,r2,nf,nE,Tt0,br_bv,cr_cv,ARv,sweepTE):
        if isinstance(br_bv,(float,int)) and isinstance(self.ARv,(float,int)) and isinstance(self.sweepTE,(float,int)):
            y_LE_0 = 0
            x_LE_0 = 0
            c_root = self.final_VT_rudder(r1,r2,nf,nE,Tt0,br_bv,cr_cv,ARv=self.ARv,sweepTE=self.sweepTE)[1]
            x_TE_1 = c_root
            y_TE_1 = 0
            bv = self.final_VT_rudder(r1,r2,nf,nE, Tt0,  br_bv, cr_cv,ARv=self.ARv,sweepTE=self.sweepTE)[3]
            y_TE_2 = bv
            x_TE_2 = c_root+bv/np.tan(np.pi/2-self.sweepTE)
            y_LE_3 = y_TE_2
            x_LE_3 =x_TE_2-self.taper_v*c_root
            y_up = br_bv * y_TE_2
            y_down = 0
            x1 = x_TE_1 - cr_cv * x_TE_1
            x2 = x_TE_1
            x3 = c_root + bv*br_bv/np.tan(np.pi/2-self.sweepTE)
            x4 = x3-c_root*cr_cv*0.4
            x_r = np.array([x1, x2, x3, x4, x1])
            y_r = np.array([y_down, y_down, y_up, y_up, y_down])
            x_points = np.array([x_LE_0, x_TE_1, x_TE_2, x_LE_3, 0])
            y_points = np.array([y_LE_0, y_TE_1, y_TE_2, y_LE_3, 0])
            Sv_estimate = (x_TE_1+(x_TE_1-x_LE_3))/2*y_TE_2
            plt.plot(x_points, y_points, label="Vertical tail")
            plt.plot(x_r, y_r, label="Rudder")
            x = np.linspace(c_root,c_root*1.5)
            plt.legend()
            plt.show()
        elif not isinstance(self.ARv,(float,int)) and isinstance(br_bv,(float,int)) and isinstance(self.sweepTE,(float,int)):
            Svstab = self.VT_stability(ARv,sweepTE)
            Svcontrol = self.VT_controllability(r1,r2,nf,nE,Tt0,br_bv,cr_cv,ARv,sweepTE)
            bv = self.final_VT_rudder(r1,r2,nf,nE,Tt0,br_bv,cr_cv,ARv=self.ARv,sweepTE=self.sweepTE)[3]
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
        elif isinstance(self.ARv,(float,int)) and isinstance(br_bv,(float,int)) and not isinstance(self.sweepTE,(float,int)):
            Svstab = self.VT_stability(ARv,sweepTE)
            Svcontrol = self.VT_controllability(r1,r2,nf,nE,Tt0,br_bv,cr_cv,ARv,sweepTE)
            # bv = self.final_VT_rudder(nE, Tt0, yE, br_bv, cr_cv)[3]
            # cbar = plt.colorbar(cp, orientation="horizontal")
            # cbar.set_label(r"$S_v$")
            # plt.ylabel(r"$b_r/b_v$ [-]", fontsize=12)
            # plt.xlabel(r"$c_r/c_v$ [-]", fontsize=12)
            # plt.show()
            Sv_estimate = None
            plt.plot(self.sweepTE*180/np.pi, Svstab, label="Stability Curve")
            plt.plot(self.sweepTE*180/np.pi, Svcontrol, label="Controllability for OEI condition")
            plt.xlabel(r"$\Lambda_{v_{TE}}$ [deg]")
            plt.ylabel(r"$S_v [m^2]$")
            plt.legend()
            plt.show()
            # plt.plot(self.sweepTE, bv)
            # plt.xlabel(r"$AR_v [-]$")
            # plt.ylabel(r"$b_v [m]$")
            # plt.show()
        elif not isinstance(self.ARv, (float, int)) and isinstance(br_bv, (float, int)) and not isinstance(self.sweepTE,(float, int)):
            X, Y = np.meshgrid(ARv, sweepTE)
            # r1, r2, nf, nE, Tt0, br_bv, cr_cv, ARv = self.ARv, sweepTE = self.sweepTE
            Z = self.final_VT_rudder(r1,r2,nf,nE, Tt0, br_bv, cr_cv,X,Y)[0]
            fig, ax = plt.subplots(1, 1)
            Sv_estimate = None
            # ax.add_artist(ab)
            # levels = [0,0.1,1,1.]
            cp = ax.contourf(X, Y*180/np.pi, Z, cmap='coolwarm', levels=50)
            # Svstab = ax.contour(X, Y, Z, [self.VT_stability(self.ARv,self.sweepTE)], colors=["k"])
            # print("Sv_stability = ",self.VT_stability(lv))
            # plt.clabel(Svstab,fmt=r"Min. :  %.1f"%(self.VT_stability(lv)))
            cbar = plt.colorbar(cp, orientation="horizontal")
            cbar.set_label(r"$S_v$ $[m^2]$")
            plt.ylabel(r"$\Lambda_{v_{TE}}$ [deg]", fontsize=12)
            plt.xlabel(r"$AR_v$ [-]", fontsize=12)
            plt.show()
            bv = self.final_VT_rudder(r1,r2,nf,nE, Tt0, br_bv, cr_cv, ARv=X, sweepTE=Y)[3]
            fig, ax = plt.subplots(1, 1)
            Sv_estimate = None
            # ax.add_artist(ab)
            # levels = [0,0.1,1,1.]
            cp = ax.contourf(X, Y * 180 / np.pi, bv, cmap='coolwarm', levels=50)
            # Svstab = ax.contour(X, Y, Z, [self.VT_stability(self.ARv,self.sweepTE)], colors=["k"])
            # print("Sv_stability = ",self.VT_stability(lv))
            # plt.clabel(Svstab,fmt=r"Min. :  %.1f"%(self.VT_stability(lv)))
            cbar = plt.colorbar(cp, orientation="horizontal")
            cbar.set_label(r"$b_v$ $[m]$")
            plt.ylabel(r"$\Lambda_{v_{TE}}$ [deg]", fontsize=12)
            plt.xlabel(r"$AR_v$ [-]", fontsize=12)
            plt.show()
        else:
            X, Y = np.meshgrid(cr_cv, br_bv)
            Z = self.VT_controllability(r1,r2,nf,nE,Tt0,Y,X,ARv,sweepTE)
            fig, ax = plt.subplots(1, 1)
            Sv_estimate = None
            # ax.add_artist(ab)
            # levels = [0,0.1,1,1.]
            cp = ax.contourf(X, Y, Z, cmap='coolwarm',levels=25)
            Svstab = ax.contour(X, Y, Z, [self.VT_stability(ARv,sweepTE)], colors=["k"])
            print("Sv_stability = ",self.VT_stability(self.ARv, self.sweepTE))
            # plt.clabel(Svstab,fmt=r"Min. :  %.3f"%(self.VT_stability(self.ARv,self.sweepTE)))
            cbar = plt.colorbar(cp, orientation="horizontal")
            cbar.set_label(r"$S_v$ $[m^2]$")
            plt.ylabel(r"$b_r/b_v$ [-]", fontsize=12)
            plt.xlabel(r"$c_r/c_v$ [-]", fontsize=12)
            plt.show()
            X, Y = np.meshgrid(cr_cv, br_bv)
            Sv = self.VT_controllability(r1,r2,nf,nE, Tt0, Y, X, ARv, sweepTE)
            Z = np.sqrt(self.ARv*Sv)
            fig, ax = plt.subplots(1, 1)
            Sv_estimate = None
            # ax.add_artist(ab)
            # levels = [0,0.1,1,1.]
            cp = ax.contourf(X, Y, Z, cmap='coolwarm', levels=25)
            Svstab = ax.contour(X, Y, Z, [np.sqrt(self.VT_stability(ARv, sweepTE)*self.ARv)], colors=["k"])
            # plt.clabel(Svstab, fmt=r"Min. :  %.3f" % (np.sqrt(self.VT_stability(ARv, sweepTE)*self.ARv)))
            cbar = plt.colorbar(cp, orientation="horizontal")
            cbar.set_label(r"$b_v$ $[m]$")
            plt.ylabel(r"$b_r/b_v$ [-]", fontsize=12)
            plt.xlabel(r"$c_r/c_v$ [-]", fontsize=12)
            plt.show()
        return Sv_estimate




