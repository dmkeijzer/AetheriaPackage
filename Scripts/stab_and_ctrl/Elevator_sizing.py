import numpy as np
from matplotlib import pyplot as plt
from stab_and_ctrl.Aileron_Sizing import Control_surface
from matplotlib import colors as mc
from Preliminary_Lift.Airfoil_analysis import Cm_ac

class Elevator_sizing:
    def __init__(self,W,h,xcg,zcg, dy, lfus,hfus,wfus,V0,Vstall,CD0,theta0,CLfwd,CLrear,
                 CLafwd,CLarear, Clafwd,Clarear, Cd0fwd, Cd0rear,
                 Sfwd,Srear,Afwd,Arear,Lambda_c4_fwd,Lambda_c4_rear,cfwd,crear,bfwd,brear,taper,dCLfwd,taper_e):
        self.W = W         # Weight [N]
        self.h = h     # Height [m]
        self.lfus = lfus # Length of the fuselage
        self.hfus = hfus # Height of the fuselage [m]
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
        self.th0 = theta0  # Initial pitch angle [rad]
        self.V0 = V0       # Initial speed [m/s]
        # self.M0 = M0       # Initial mach number [-]
        # self.Re = self.rho*self.V0*self.lfus/self.mu
        self.CLafwd, self.CLarear = CLafwd, CLarear # Wing lift curve slopes for both wings [1/rad]
        self.Cmacfwd, self.Cmacrear = Cm_ac(self.Sweepc4fwd,self.Afwd)[0],Cm_ac(self.Sweepc4rear,self.Arear)[0]
        self.CD0 = CD0 # C_D_0 of forward wing
        self.xacfwd = 0.25*self.cfwd
        self.xacrear = self.lfus - (1 - 0.25) * self.crear
        # self.de_da = self.deps_da(self.Sweepc4fwd,self.bfwd,self.lh(),self.hfus,self.Afwd,self.CLafwd)
        self.taper_v = 0.4
        self.Vs = Vstall # Stall speed [m/s]
        self.Vmc = 1.2*self.Vs # Minimum controllable speed [m/s]
        self.xcg = xcg
        self.c = self.Sfwd/self.S*self.cfwd+self.Srear/self.S*self.crear
        self.dCLfwd = dCLfwd
        # self.b1 = b1
        # self.b2  = b2
        self.taper_e = taper_e
        # self.Sa_S = Sa_S
        self.Clafwd, self.Clarear, self.Cd0fwd, self.Cd0rear = Clafwd, Clarear, Cd0fwd,Cd0rear
        self.eta_rear = 1.0
        self.dy = dy
        self.zcg = zcg

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
    def tau_e(self,Se_S):
        """
        Inputs:
        :param Se_S: Elevator surface to wing ratio [-]
        :return: Elevator Effectiveness [-]
        """
        x = Se_S
        tau_a = -6.624*x**4+12.07*x**3-8.292*x**2+3.295*x+0.004942
        return tau_a
    def dCLfwd_f(self,Se_S,be_b,de_max):
        dCL = self.tau_e(Se_S) * self.CLafwd * be_b * de_max * np.pi / 180/100
        return dCL
    def dCLrear(self,Se_S,be_b,de_max):
        dCL = -self.tau_e(Se_S) * self.CLarear * be_b * de_max * np.pi / 180 / 100
        return dCL
    def Cm(self,Se_S,be_b,de_max):
        CDfwd = self.CD0 + self.CLfwd**2/(np.pi*0.65*self.Afwd)
        CDrear = self.CD0 + self.CLrear ** 2 / (np.pi * 0.65 * self.Arear)
        Cm = -CDfwd*(self.zcg-self.dy)*self.Sfwd/(self.S*self.c)+\
             (self.CLfwd+self.dCLfwd_f(Se_S,be_b,de_max))*(self.xcg-self.xacfwd)*self.Sfwd/(self.S*self.c) + \
               CDrear*(self.hfus-self.zcg)*self.Srear/(self.S*self.c)*self.eta_rear-\
             (self.CLrear+self.dCLrear(Se_S,be_b,de_max))*(self.xacrear-self.xcg)*self.Srear/(self.S*self.c)*self.eta_rear+\
             self.Cmacfwd*self.Sfwd*self.cfwd/(self.S*self.c)+self.Cmacrear*self.Srear*self.crear/(self.S*self.c)*self.eta_rear
        return Cm
    def plotting(self,Se_S,be_b,ce_c, de_max):
        """
        Inputs:
        :param Se_S: Elevator surface to wing ratio [-]
        :param de_max: Maximum elevator deflection
        :return: Plots
        """
        mindCL = self.dCLfwd
        if isinstance(be_b,float) and not isinstance(Se_S,float):
            mindCL = np.ones(len(be_b))*mindCL
            dCL_array = self.tau_e(Se_S)*self.CLafwd*be_b*de_max*np.pi/180/100
            plt.plot(be_b,mindCL,label=r"Minimum value required $\Delta C_{L_{fwd}}$")
            plt.plot(be_b,dCL_array,label=r"$\Delta C_{L_{fwd}}$")
            plt.xlabel(r"$b_e/b_{fwd} [m]$",fontsize=14)
            plt.ylabel(r"$\Delta C_{L_{fwd}}$",fontsize=14)
            # plt.vlines(b1, min(Clda_array),max(Clda_array),"r",label=r"Smallest limit set by $b_1$")
            plt.ylim(min(dCL_array))
            plt.xlim(min(be_b))
            plt.legend()
            plt.show()

        elif isinstance(Se_S,(float,int)) and isinstance(be_b,(float,int)):
            x_1 = 0
            y_1 = 0
            x_2 = self.bfwd / 2
            y_2 = np.tan(self.Sweep(self.Afwd, 0, 100, 25)) * self.bfwd / 2
            y_2 = abs(y_2)
            x_4 = 0
            y_4 = self.cfwd * 3 / 2 * (1 + self.taper) / (1 + self.taper + self.taper ** 2)
            x_p_3 = np.tan(self.Sweep(self.Afwd, 0, 0, 25)) * self.bfwd / 2
            x_3 = self.bfwd / 2
            y_3 = abs(y_4 - x_p_3)
            x_points = [x_1, x_2, x_3, x_4, x_1]
            y_points = [y_1, y_2, y_3, y_4, y_1]
            # xa_1 = b1/100*self.bfwd/2
            xa_1 = 1/2
            ya_1= abs(xa_1*np.tan(self.Sweep(self.Afwd, 0, 100, 25)))
            xa_2 = be_b*self.bfwd/2/100
            ya_2 = abs(xa_2*np.tan(self.Sweep(self.Afwd, 0, 100, 25)))
            #### Aileron geometry ####
            ce = Se_S*self.Sfwd/(be_b*self.bfwd/100)
            # print("c_a = ",ca)
            ca_r = ce * 2/(1+self.taper_e)
            # print("ca_root = ",ca_r)
            ca_t = ca_r*self.taper_e
            xa_3 = xa_2
            ya_3 = ca_t+ya_2
            xa_4 = xa_1
            ya_4 = ya_1+ca_r
            xa_points= [xa_1,xa_2,xa_3,xa_4,xa_1]
            ya_points= [ya_1,ya_2,ya_3,ya_4,ya_1]
            plt.plot(x_points, y_points,label="Forward Wing")
            plt.plot(xa_points,ya_points,"r",label="Elevator")
            plt.legend()
            plt.show()
        else:
            # aileron =Control_surface(self.V0,self.Vs,self.CLfwd,self.CLrear,
            #      self.CLafwd,self.CLarear, self.Clafwd,self.Clarear,self.Cd0fwd,self.Cd0rear,
            #      self.Sfwd,self.Srear,self.Afwd,self.Arear,self.cfwd,self.crear,self.bfwd,self.brear,self.taper,self.eta_rear)
            # ca_t  = aileron.plotting(self.Sa_S,self.b1,self.b2,rear=True)[0]
            # Se_S_geo = self.b2/100/2*self.bfwd*(ca_t+ca_t/self.taper_a**2)/self.Sfwd
            X, Y = np.meshgrid(be_b, Se_S)
            Z = self.Cm(Y,X,de_max)
            fig, ax = plt.subplots(1, 1)
            # ax.add_artist(ab)
            # levels = [0,0.1,1,1.]
            cp = ax.contourf(X, Y, Z, cmap='coolwarm',levels=20)
            minimum = ax.contour(X, Y, Z, [0], colors=["k"])
            plt.clabel(minimum,fmt="Trim condition")
            cbar = plt.colorbar(cp, orientation="horizontal")
            cbar.set_label(r"$C_{m}$ [-]")
            # plt.hlines(Se_S_geo,min(be_b),max(be_b),label="Required value from aileron")
            plt.ylabel(r"$S_e/S_{i}$ [-]", fontsize=12)
            plt.xlabel(r"$b_e$ [$\% b_{i}$]", fontsize=12)
            # plt.legend()
            # plt.vlines(b1,min(Sa_S),max(Sa_S),"r",label=r"Smallest limit set by $b_1$")
            plt.show()
            # ce_c = Se_S*self.Sfwd/(be_b/100)/self.bfwd/self.cfwd
            X, Y = np.meshgrid(be_b, ce_c)
            Se_S = Y*be_b/100*(self.bfwd/self.Sfwd)*self.cfwd
            Z = self.Cm(Se_S, X, de_max)
            fig, ax = plt.subplots(1, 1)
            # ax.add_artist(ab)
            # levels = [0,0.1,1,1.]
            cp = ax.contourf(X, Y, Z, cmap='coolwarm', levels=20)
            minimum = ax.contour(X, Y, Z, [0], colors=["k"])
            plt.clabel(minimum, fmt="Trim condition")
            cbar = plt.colorbar(cp, orientation="horizontal")
            cbar.set_label(r"$C_m$ [-]")
            # plt.hlines(Se_S_geo,min(be_b),max(be_b),label="Required value from aileron")
            plt.ylabel(r"$\bar{c}_e/\bar{c}_{i}$ [-]", fontsize=12)
            plt.xlabel(r"$b_e$ [$\% b_{i}$]", fontsize=12)
            # plt.legend()
            # plt.vlines(b1,min(Sa_S),max(Sa_S),"r",label=r"Smallest limit set by $b_1$")
            plt.show()




