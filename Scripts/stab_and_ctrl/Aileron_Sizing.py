import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mc

class Control_surface:
    def __init__(self,V0,Vstall,CLfwd,CLrear,
                 CLafwd,CLarear, Clafwd,Clarear,Cd0fwd,Cd0rear,
                 Sfwd,Srear,Afwd,Arear,cfwd,crear,bfwd,brear,taper,eta_rear):
        # self.lfus = lfus # Length of the fuselage
        # self.hsus = hfus # Height of the fuselage [m]
        # self.wfus = wfus # Maximum width of the fuselage [m]
        self.Srear = Srear # Rear wing area [m^2]
        self.Sfwd = Sfwd   # Forward wing area [m^2]
        self.S = Srear+Sfwd # Aircraft wing area [m^2]
        self.cfwd = cfwd         # Average chord [m]
        self.crear = crear  # Average chord [m]
        self.bfwd = bfwd         # Wing span [m]
        self.brear = brear # Wing span [m]
        self.taper = taper # Wing taper ratio [-]
        self.Clafwd,self.Clarear = Clafwd,Clarear # Airfoil lift curve [1/rad]
        self.Cd0fwd,self.Cd0rear = Cd0fwd,Cd0rear # Airfoil zero drag coefficient [-]
        self.CLfwd,self.CLrear  = CLfwd,CLrear # DESIGN FOR CRUISE Lift coefficients [-]
        self.Afwd, self.Arear = Afwd, Arear # Aspect ratio of both wings [-]
        # self.Sweepc2fwd = Lambda_c2_fwd # Sweep at c/2 [rad]
        # self.Sweepc2rear = Lambda_c2_rear # Sweep at c/2 [rad]
        # self.Sweepc4fwd = self.Sweep(Afwd,self.Sweepc2fwd,25,50)
        # self.Sweepc4rear = self.Sweep(Arear, self.Sweepc2rear, 25, 50)
        self.V0 = V0       # Initial speed [m/s]
        self.CLafwd, self.CLarear = CLafwd, CLarear # Wing lift curve slopes for both wings [1/rad]
        # self.xacfwd = 0.25*self.cfwd
        # self.xacrear = self.lfus - (1 - 0.25) * self.crear
        self.Vs = Vstall # Stall speed [m/s]
        self.Vmc = 1.2*self.Vs # Minimum controllable speed [m/s]
        # self.xcg = xcg
        self.c = self.Sfwd/self.S*self.cfwd+self.Srear/self.S*self.crear
        self.taper_a = 0.45
        self.eta_rear = eta_rear

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
        tanSweep_n = tanSweep_m -4/(AR)*(n-m)/100*(1-self.taper)/(1+self.taper)
        return np.arctan(tanSweep_n)

    def tau_a(self,Sa_S):
        """
        Inputs:
        :param Sa_S: Aileron surface to wing ratio [-]
        :return: Aileron Effectiveness [-]
        """
        x = Sa_S
        tau_a = -6.624*x**4+12.07*x**3-8.292*x**2+3.295*x+0.004942
        return tau_a

    def Clda(self,Sa_S,b1,b2,rear):
        """
        Input:
        :param b2: Outer distance of the aileron [m]
        :return: Control derivative C_l_da [1/rad]
        """
        b =  np.sqrt(0.5*(self.Srear/self.S*self.Arear+self.Sfwd/self.S*self.Afwd)*self.S)
        b_1fwd = b1*self.bfwd / 2 / 100
        b_2fwd = b2*self.bfwd / 2 / 100
        b_1rear = b1 * self.brear / 2 / 100
        b_2rear = b2 * self.brear / 2 / 100
        c_r_fwd = self.cfwd*3/2*(1+self.taper)/(1+self.taper+self.taper**2)
        c_r_rear = self.crear * 3 / 2 * (1 + self.taper) / (1 + self.taper + self.taper ** 2)
        Cldafwd = -self.CLafwd*self.tau_a(Sa_S)*c_r_fwd/(self.Sfwd*self.bfwd)*\
               (0.5*(b_2fwd**2-b_1fwd**2) + 2*(self.taper-1)/(3*self.bfwd)*(b_2fwd**3-b_1fwd**3))
        Cldafwd *= self.Sfwd*self.bfwd/(self.S*b)
        if rear==False:
            cc =0
        else:
            cc=1
        Cldarear = -self.CLarear*self.tau_a(Sa_S)*c_r_rear/(self.Srear*self.brear)*\
               (0.5*(b_2rear**2-b_1rear**2) + 2*(self.taper-1)/(3*self.brear)*(b_2rear**3-b_1rear**3))*self.eta_rear
        Cldarear *= self.Srear * self.brear / (self.S * b)*cc
        Clda = Cldafwd+Cldarear
        return Clda

    def Clp(self):
        b =  np.sqrt(0.5*(self.Srear/self.S*self.Arear+self.Sfwd/self.S*self.Afwd)*self.S)
        c_r_fwd = self.cfwd * 3 / 2 * (1 + self.taper) / (1 + self.taper + self.taper ** 2)
        c_r_rear =  self.crear*3/2*(1+self.taper)/(1+self.taper+self.taper**2)
        Clp_fwd =-(self.Clafwd+self.Cd0fwd)*c_r_fwd*self.bfwd/(24*self.Sfwd)*(1+3*self.taper)*(1-1.38/self.bfwd)
        Clp_rear = -(self.Clarear + self.Cd0rear) * c_r_rear * self.brear/(24 * self.Srear)*(1+3*self.taper)*self.eta_rear**(1-1.38/self.bfwd)
        Clp = Clp_fwd*self.Sfwd*self.bfwd/(self.S*b)+Clp_rear*self.Srear*self.brear/(self.S*b)
        return Clp

    def plotting(self,Sa_S,b1,b2,Se_S, be_b, rear):
        if isinstance(Sa_S,(float,int)) and not isinstance(b2,(float,int)):
            da_max = -30*np.pi/180
            dphi_dt = 60*np.pi/180/1.3*0.95
            minClda = -(dphi_dt)*self.Clp()*max(self.bfwd,self.brear)/(2*self.Vmc*da_max)
            minClda = np.ones(len(b2))*minClda
            Clda_array = self.Clda(Sa_S,b1,b2,rear)
            plt.plot(b2,minClda,label=r"Minimum value required $C_{l_{\delta_a}}$")
            plt.plot(b2,Clda_array,label=r"$C_{l_{\delta_a}}(b_2)$")
            plt.xlabel(r"$b_2 [m]$",fontsize=14)
            plt.ylabel(r"$C_{l_{\delta_a}} [1/rad]$",fontsize=14)
            # plt.vlines(b1, min(Clda_array),max(Clda_array),"r",label=r"Smallest limit set by $b_1$")
            plt.ylim(min(Clda_array))
            plt.xlim(min(b2))
            plt.legend()
            plt.show()

        elif isinstance(Sa_S,(float,int)) and isinstance(b1,(float,int)):
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
            xa_1 = b1/100*self.bfwd/2
            ya_1= abs(b1/100*self.bfwd/2*np.tan(self.Sweep(self.Afwd, 0, 100, 25)))
            xa_2 = b2/100*self.bfwd/2
            ya_2 = abs(b2/100*self.bfwd/2*np.tan(self.Sweep(self.Afwd, 0, 100, 25)))
            #### Aileron geometry ####
            # ba = (b2-b1)/100*self.bfwd/2*2
            # ca = Sa_S*self.Sfwd/ba
            # # print("c_a = ",ca)
            # ca_r = ca * 2/(1+self.taper_a)
            # # print("ca_root = ",ca_r)
            # ca_t = ca_r*self.taper_a
            ce = Se_S * self.Sfwd / (be_b * self.bfwd / 100)
            ce_r = ce * 2 / (1 + 0.45)
            ce_t = ce_r * 0.45
            ca_t = ce_t
            ce_t2 = ce_r * (1 - 2 * (1 - 0.45) * (b2 * self.bfwd / 2 / 100 - 0.5) / (be_b * (self.bfwd) / 100))
            print("ce_t = ",ce_t, "ce_t_with equation", ce_t2)
            ca_r = ce_r * (1 - 2 * (1 - 0.45) * (b1 * self.bfwd / 2/100-0.5) / (be_b * (self.bfwd)/100))
            ba = (b2 - b1) / 100 * self.bfwd / 2*2
            xa_3 = xa_2
            ya_3 = ca_t+ya_2
            xa_4 = xa_1
            ya_4 = ya_1+ca_r
            xa_points= [xa_1,xa_2,xa_3,xa_4,xa_1]
            ya_points= [ya_1,ya_2,ya_3,ya_4,ya_1]
            ## Plot elevator ##
            xe_1 = 1 / 2
            ye_1 = abs(xe_1 * np.tan(self.Sweep(self.Afwd, 0, 100, 25)))
            xe_2 = 99 * self.bfwd / 2 / 100
            ye_2 = abs(xe_2 * np.tan(self.Sweep(self.Afwd, 0, 100, 25)))
            #### Aileron geometry ####
            ce = Se_S * self.Sfwd / (be_b * self.bfwd / 100)
            # print("c_a = ",ca)
            ce_r = ce * 2 / (1 + 0.45)
            # print("ca_root = ",ca_r)
            ce_t = ce_r *0.45
            xe_3 = xe_2
            ye_3 = ce_t + ye_2
            xe_4 = xe_1
            ye_4 = ye_1 + ce_r
            xe_points = [xe_1, xe_2, xe_3, xe_4, xe_1]
            ye_points = [ye_1, ye_2, ye_3, ye_4, ye_1]
            plt.plot(x_points, y_points,label="Forward Wing")
            plt.plot(xa_points,ya_points,"g", label="Aileron")
            plt.plot(xe_points,ye_points, "r", label="Elevator")
            plt.legend()
            plt.show()
        else:
            ce = Se_S * self.Sfwd / (be_b * self.bfwd / 100)
            # print("c_a = ",ca)
            ce_r = ce * 2 / (1 + 0.45)
            # print("ce_r = %.3f"%(ce_r))
            # print("ca_root = ",ca_r)
            ce_t = ce_r * 0.45
            # print("ce_t = %.3f" % (ce_t))
            ca_t = ce_t
            ca_r = ce_r*(1-2*(1-0.45)*(b1*self.bfwd/2/100-0.5)/(be_b*self.bfwd/100))
            # print("be = %.5f"%(be_b*self.bfwd/100))
            # print("ca_r = %.5f"%(ca_r))

            ba = (b2 - b1) / 100 * self.bfwd / 2*2
            Sa_S_geo = ba*(ca_t+ca_r)/2/self.Sfwd
            # print("c_a = ",ca)
            # print("ca_root = ",ca_r)
            da_max = -30 * np.pi / 180
            dphi_dt_1 = 10*0.3048/max(self.brear,self.bfwd)*2
            dphi_dt = 60 * np.pi / 180 / 1.3
            p_min=dphi_dt
            minClda = -(dphi_dt) * self.Clp() * max(self.bfwd, self.brear) / (2 * self.Vmc * da_max)
            print("minClda = %.5f"%(minClda))
            X, Y = np.meshgrid(b1, Sa_S)
            Cl_da=  self.Clda(Y,X, b2,rear)
            Z = -2*self.Vmc/max(self.bfwd, self.brear) *Cl_da/self.Clp()*da_max
            fig, ax = plt.subplots(1, 1)
            # ax.add_artist(ab)
            # levels = [0,0.1,1,1.]
            cp = ax.contourf(X, Y, Z, cmap='RdGy',levels=20)
            minimum = ax.contour(X, Y, Z, [p_min], colors=["k"])
            plt.clabel(minimum,fmt="Roll requirement")
            cbar = plt.colorbar(cp, orientation="horizontal")
            cbar.set_label(r"$p$ [$rad/s$]")
            plt.plot(b1, Sa_S_geo, label="Constraint from elevator")
            plt.ylabel(r"$S_a/S_{i}$ [-]", fontsize=12)
            plt.xlabel(r"$b_1$ [$\% b_{i}/2$]", fontsize=12)
            plt.ylim(min(Sa_S), max(Sa_S))
            plt.legend()
            # plt.vlines(b1,min(Sa_S),max(Sa_S),"r",label=r"Smallest limit set by $b_1$")
            plt.show()
        return ca_t, ca_r, b1, b2




