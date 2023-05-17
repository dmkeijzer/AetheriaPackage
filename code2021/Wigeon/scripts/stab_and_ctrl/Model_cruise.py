from control import matlab
import numpy as np
import matplotlib.pyplot as plt
from math import *
import scipy.integrate as sp

"""
Construction of the state-space systems for symmetric and asymmetric motion,
as well as functions to simulate the eigenmotions.

author: Michal Cuadrat-Grzybowski
"""


class Aircraft:
    def __init__(self,W,h,S,c,b,V,theta0,alpha0,q0,b0,phi0,p0,r0,Iyy,Ixx,Izz,Ixz,C_X,C_Z,C_m,C_L,C_Y,C_l,C_n,Ka):
        self.W = W     # Weight [N]
        self.h = h     # Height [m]
        self.S = S     # Total wing area [m^2]
        self.c = c     # Total mean aerodynamic chord [m]
        self.b = b     # Maximum wing span [m]
        self.Ka = Ka
        #### Initial conditons ####
        self.th0 = theta0 * np.pi / 180
        self.alpha0 = alpha0*np.pi/180
        self.q0 = q0
        self.beta0 = b0* np.pi / 180
        self.phi0 = phi0* np.pi / 180
        self.p0 = p0
        self.r0= r0
        ####### Inertia ########
        self.Kyy2 = Iyy/(self.W/9.80665*self.c**2)
        self.Kxx2 = Ixx/(self.W/9.80665*self.b**2)
        self.Kzz2 = Izz/(self.W/9.80665*self.b**2)
        self.Kxz = Ixz/(self.W/9.80665*self.b**2)
        self.sym_sys = self.compute_sym_sys(V,self.Kyy2, C_X,C_Z,C_m)
        self.asym_sys = self.compute_asym_sys(V,self.Kxx2,self.Kzz2,self.Kxz, C_L,C_Y,C_l,C_n)
        self.sym_sys_tuned = self.alpha_tf(self.Ka)
        self.eig_s = self.get_eigenvalues(self.sym_sys,V,"sym")
        self.eig_a = self.get_eigenvalues(self.asym_sys,V,"asym")
        self.eig_s_tuned = self.get_eigenvalues(self.sym_sys_tuned,V,"sym")
        self.prop_s = self.get_period_prop(self.sym_sys,V,"sym")
        self.prop_a = self.get_period_prop_a(self.asym_sys, V, "asym")
        self.muc = self.get_muc(self.W,self.h,self.S,self.c)
        self.mub = self.get_mub(self.W, self.h, self.S, self.b)
        print("mu_c = %.4f , mu_b = %.4f "%(self.muc,self.mub))

    def plot_results(self, V):
        T = np.linspace(0, 250, 2500)

        #### Compute Symmetric System Eigenvalues ####
        print("-----------Symmetric Case (Non-tuned)-----------")
        lc = '\u03BB_c = '
        print("Eigenvalues for Symmetric Motion", lc,self.eig_s)
        print("Properties of motion:", self.prop_s[0],self.prop_s[1])

        #### Compute Symmetric System Eigenvalues ####
        print("-----------Symmetric Case (Tuned)-----------")
        lc = '\u03BB_c = '
        print("Eigenvalues for Symmetric Motion", lc, self.eig_s_tuned)
        # print("Properties of motion:", self.prop_s[0], self.prop_s[1])

        ############### Compute Phugoid & Plotting ##############
        T_p = np.linspace(0, 500, 1000)
        delta_e = self.pulse(T_p, 1,16)*radians(0.5)
        u_motion, alpha_motion, theta_motion, pitch_motion = \
            self.phugoid(T_p, delta_e, V,sys=self.sym_sys)

        fig,axs = plt.subplots(2,2, figsize=(7,4), constrained_layout=True)
        axs[0,0].plot(T_p,u_motion)
        axs[0,0].set_xlabel(r"$t$ [s]", fontsize=12)
        axs[0, 0].set_ylabel(r"$u$ [m/s]", fontsize=12)


        axs[0, 1].plot(T_p, alpha_motion)
        axs[0, 1].set_xlabel(r"$t$ [s]", fontsize=12)
        axs[0, 1].set_ylabel(r"$\alpha$ [rad]", fontsize=12)


        axs[1, 0].plot(T_p, theta_motion)
        axs[1, 0].set_xlabel(r"$t$ [s]", fontsize=12)
        axs[1, 0].set_ylabel(r"$\theta$ [rad]", fontsize=12)


        axs[1, 1].plot(T_p, pitch_motion)
        axs[1, 1].set_xlabel(r"$t$ [s]", fontsize=12)
        axs[1, 1].set_ylabel(r"$q$ [rad/s]", fontsize=12)
        plt.suptitle("Phugoid Motion", fontsize=16)
        plt.show()

        ############### Compute Short Period & Plotting ##############
        T_sp = np.linspace(0, 20, 100)
        delta_e = self.pulse(T_sp,1,1.5)*radians(0.5)
        u_motion, alpha_motion, theta_motion, pitch_motion = \
            self.short_period(T_sp, delta_e, V,sys=self.sym_sys)

        fig,axs = plt.subplots(2,2, figsize=(7,4), constrained_layout=True)
        axs[0,0].plot(T_sp,u_motion)
        axs[0,0].set_xlabel(r"$t$ [s]", fontsize=12)
        axs[0, 0].set_ylabel(r"$u$ [m/s]", fontsize=12)


        axs[0, 1].plot(T_sp, alpha_motion)
        axs[0, 1].set_xlabel(r"$t$ [s]", fontsize=12)
        axs[0, 1].set_ylabel(r"$\alpha$ [rad]", fontsize=12)


        axs[1, 0].plot(T_sp, theta_motion)
        axs[1, 0].set_xlabel(r"$t$ [s]", fontsize=12)
        axs[1, 0].set_ylabel(r"$\theta$ [rad]", fontsize=12)


        axs[1, 1].plot(T_sp, pitch_motion)
        axs[1, 1].set_xlabel(r"$t$ [s]", fontsize=12)
        axs[1, 1].set_ylabel(r"$q$ [rad/s]", fontsize=12)
        plt.suptitle("Short Period", fontsize=16)
        plt.show()

        ######## Asymmetric System ##########
        print("-------Asymmetric Case (Non-tuned) ---------")
        lb = '\u03BB_b = '
        print(r"Eigenvalues for Asymmetric Motion ", lb, self.eig_a)
        print("Properties of motion:", self.prop_a)

        ############### Compute Dutch roll & Plotting ##############
        delta_r = self.pulse(T,2,2.5)*radians(0.2)
        beta_motion, phi_motion, p_motion, r_motion, extra_var = self.dutch_roll(T, delta_r, V)
        fig, axs = plt.subplots(2, 2, figsize=(7, 4), constrained_layout=True)
        axs[0, 0].plot(T, beta_motion)
        axs[0, 0].set_xlabel(r"$t$ [s]", fontsize=12)
        axs[0, 0].set_ylabel(r"$\beta$ [rad]", fontsize=12)

        axs[0, 1].plot(T, phi_motion)
        axs[0, 1].set_xlabel(r"$t$ [s]", fontsize=12)
        axs[0, 1].set_ylabel(r"$\phi$ [rad]", fontsize=12)

        axs[1, 0].plot(T, p_motion)
        axs[1, 0].set_xlabel(r"$t$ [s]", fontsize=12)
        axs[1, 0].set_ylabel(r"$p$ [rad/s]", fontsize=12)

        axs[1, 1].plot(T, r_motion)
        axs[1, 1].set_xlabel(r"$t$ [s]", fontsize=12)
        axs[1, 1].set_ylabel(r"$r$ [rad/s]", fontsize=12)
        plt.suptitle("Dutch Roll", fontsize=16)
        plt.show()

        #### Plot Heading Psi ####

        # T_h = T
        # psi = self.yaw(T_h,delta_r,V,"dr")
        # x_chi = self.heading(T_h,delta_r,V,"dr")
        # T_h = np.delete(T_h, 0)
        # fig, axs = plt.subplots(2, 1, figsize=(7, 4), constrained_layout=True)
        # axs[0].plot(T_h, psi)
        # axs[0].set_xlabel(r"$t$ [s]", fontsize=12)
        # axs[0].set_ylabel(r"$\psi$ [rad]", fontsize=12)
        #
        # axs[1].plot(T_h, x_chi)
        # axs[1].set_xlabel(r"$t$ [s]", fontsize=12)
        # axs[1].set_ylabel(r"$\chi$ [rad]", fontsize=12)
        # plt.suptitle("Yaw and Heading angles Dutch Roll")
        # plt.show()

        ############### Compute Aperiodic Roll & Plotting ##############
        delta_a = self.pulse(T,1,15)*radians(0.2)
        beta_motion, phi_motion, p_motion, r_motion, extra_var = self.ap_roll(T, delta_a, V)

        fig, axs = plt.subplots(2, 2, figsize=(7, 4), constrained_layout=True)
        axs[0, 0].plot(T, beta_motion)
        axs[0, 0].set_xlabel(r"$t$ [s]", fontsize=12)
        axs[0, 0].set_ylabel(r"$\beta$ [rad]", fontsize=12)

        axs[0, 1].plot(T, phi_motion)
        axs[0, 1].set_xlabel(r"$t$ [s]", fontsize=12)
        axs[0, 1].set_ylabel(r"$\phi$ [rad]", fontsize=12)

        axs[1, 0].plot(T, p_motion)
        axs[1, 0].set_xlabel(r"$t$ [s]", fontsize=12)
        axs[1, 0].set_ylabel(r"$p$ [rad/s]", fontsize=12)

        axs[1, 1].plot(T, r_motion)
        axs[1, 1].set_xlabel(r"$t$ [s]", fontsize=12)
        axs[1, 1].set_ylabel(r"$r$ [rad/s]", fontsize=12)
        plt.suptitle("Aperiodic Roll", fontsize=16)
        plt.show()
        #### Plot Heading Psi, Chi  ####

        # T_h = T
        # psi = self.yaw(T_h,delta_a,V,"apr")
        # x_chi = self.heading(T_h,delta_a,V,"apr")
        # T_h = np.delete(T_h, 0)
        # fig, axs = plt.subplots(2, 1, figsize=(7, 4), constrained_layout=True)
        # axs[0].plot(T_h, psi)
        # axs[0].set_xlabel(r"$t$ [s]", fontsize=12)
        # axs[0].set_ylabel(r"$\psi$ [rad]", fontsize=12)
        #
        # axs[1].plot(T_h, x_chi)
        # axs[1].set_xlabel(r"$t$ [s]", fontsize=12)
        # axs[1].set_ylabel(r"$\chi$ [rad]", fontsize=12)
        # plt.suptitle("Yaw and Heading angles Aperiodic Roll")
        # plt.show()

        ############### Compute Spiral & Plotting ##############
        delta_a = self.pulse(T,2,2.5)*radians(0.2)
        beta_motion, phi_motion, p_motion, r_motion, extra_var = self.spiral(T, delta_a, V)

        fig, axs = plt.subplots(2, 2, figsize=(7, 4), constrained_layout=True)
        axs[0, 0].plot(T, beta_motion)
        axs[0, 0].set_xlabel(r"$t$ [s]", fontsize=12)
        axs[0, 0].set_ylabel(r"$\beta$ [rad]", fontsize=12)

        axs[0, 1].plot(T, phi_motion)
        axs[0, 1].set_xlabel(r"$t$ [s]", fontsize=12)
        axs[0, 1].set_ylabel(r"$\phi$ [rad]", fontsize=12)

        axs[1, 0].plot(T, p_motion)
        axs[1, 0].set_xlabel(r"$t$ [s]", fontsize=12)
        axs[1, 0].set_ylabel(r"$p$ [rad/s]", fontsize=12)

        axs[1, 1].plot(T, r_motion)
        axs[1, 1].set_xlabel(r"$t$ [s]", fontsize=12)
        axs[1, 1].set_ylabel(r"$r$ [rad/s]", fontsize=12)
        plt.suptitle("Aperiodic Spiral", fontsize=16)
        plt.show()

        #### Plot Heading Psi ####

        # T_h = T
        # psi = self.yaw(T_h,delta_a,V,"spiral")
        # x_chi = self.heading(T_h,delta_a,V,"spiral")
        # T_h = np.delete(T_h, 0)
        # fig, axs = plt.subplots(2, 1, figsize=(7, 4), constrained_layout=True)
        # axs[0].plot(T_h, psi)
        # axs[0].set_xlabel(r"$t$ [s]", fontsize=12)
        # axs[0].set_ylabel(r"$\psi$ [rad]", fontsize=12)
        #
        # axs[1].plot(T_h, x_chi)
        # axs[1].set_xlabel(r"$t$ [s]", fontsize=12)
        # axs[1].set_ylabel(r"$\chi$ [rad]", fontsize=12)
        # plt.suptitle("Yaw and Heading angles Spiral")
        # plt.show()

    def pulse(self,t,a,b):
        p = []
        for ti in t:
            if ti>=a and ti<b:
                p.append(1)
            else:
                p.append(0)
        return np.array(p)


    def get_rho(self,h):
        # Constant values concerning atmosphere and gravity
        rho0 = 1.2250  # air density at sea level [kg/m^3]
        lda = -0.0065  # temperature gradient in ISA [K/m]
        Temp0 = 288.15  # temperature at sea level in ISA [K]
        R = 287.05  # specific gas constant [m^2/sec^2K]
        g = 9.80665  # [m/sec^2] (gravity constant)
        return rho0 * np.power( ((1+(lda * h / Temp0))), (-((g / (lda*R)) + 1)))

    def g(self,h,Re = 6371*10**3):
        return 9.80665*(Re/(Re+h))**2

    def get_muc(self,W,h,S,c):
        rho = self.get_rho(h)
        return W/9.80665*self.g(h)/(rho*self.g(h)*S*c)

    def get_mub(self,W,h,S,b):
        rho = self.get_rho(h)
        return W/9.80665*self.g(h)/(rho*self.g(h)*S*b)

    def compute_sym_sys(self,V,Kyy2, C_X,C_Z,C_m):
        """
        Inputs:
        V: airspeed
        Kyy2: Kyy^2 normalised moment of inertia
        C_X: array containing all stability derivatives and C_X_delta related to C_X
        C_Z: array containing all stability derivatives and C_Z_delta related to C_Z
        C_m: array containing all stability derivatives and C_m_delta related to C_m (pitch)
        returns: Symmetric state-space system
        """
        mu_c = self.get_muc(self.W,self.h,self.S,self.c)
        C_x_a,C_x_a_dot, C_x_u,C_x_0,C_x_q,C_x_d = C_X
        C_z_a, C_z_a_dot, C_z_u, C_z_0, C_z_q,C_z_d = C_Z
        C_m_a,C_m_a_dot,C_m_u,C_m_q,C_m_d = C_m
        C_1 = np.array([[-2*mu_c*self.c/V,0,0,0],
                         [0,(C_z_a_dot-2*mu_c)*self.c/V,0,0],
                         [0,0,-self.c/V,0],
                         [0,C_m_a_dot*self.c/V,0,-2*mu_c*Kyy2*self.c/V]])
        C_2 = np.array([[C_x_u,C_x_a,C_z_0,C_x_q],
                         [C_z_u,C_z_a,-C_x_0,(C_z_q+2*mu_c)],
                         [0,0,0,1],
                         [C_m_u,C_m_a,0,C_m_q]])
        C_3 = np.array([[C_x_d],
                         [C_z_d],
                         [0],
                         [C_m_d]])
        A_s = -np.dot(np.linalg.inv(C_1),C_2)
        B_s = -np.dot(np.linalg.inv(C_1),C_3)
        C_s = np.identity(4)
        D_s = np.array([[0],
                         [0],
                         [0],
                         [0]])
        return matlab.ss(A_s,B_s,C_s,D_s)

    def alpha_tf(self, Ka):
        Hset = matlab.tf(self.sym_sys)
        # Hol = matlab.tf(Hset.num[i][o],Hset.den[i][o])
        # matlab.sisotool(Hol)
        # plt.show()
        Ka = np.array([[0,Ka,0,0]])
        SysCL = self.sym_sys.feedback(Ka)
        return SysCL

    def plot_open_loop(self,sys,i,o):
        Hset = matlab.tf(sys)
        Hol = matlab.tf(Hset.num[i][o], Hset.den[i][o])
        matlab.sisotool(Hol)
        plt.show()
        return



    def sym_stability_req(self,Kyy2,C_X,C_Z,C_m):
        """
        Inputs:
        :param Kyy2: Normalised moment of inertia
        :param C_X: array containing C_X derivatives
        :param C_Z: array containing C_Z derivatives
        :param C_m: array containing C_m derivatives
        :return: assesment of stability
        """
        mu_c = self.get_muc(self.W, self.h, self.S, self.c)
        C_x_a, C_x_a_dot, C_x_u, C_x_0, C_x_q, C_x_d = C_X
        C_z_a, C_z_a_dot, C_z_u, C_z_0, C_z_q, C_z_d = C_Z
        C_m_a, C_m_a_dot, C_m_u, C_m_q, C_m_d = C_m
        A = 4*mu_c**2*Kyy2*(C_z_a_dot-2*mu_c)
        B = C_m_a_dot*2*mu_c*(C_z_q+2*mu_c)-\
            C_m_q*2*mu_c*(C_z_a_dot-2*mu_c)-\
            2*mu_c*Kyy2*(C_x_u*(C_z_a_dot-2*mu_c)-2*mu_c*C_z_a)
        C = C_m_a*2*mu_c*(C_z_q+2*mu_c)-\
            C_m_a_dot*(2*mu_c*C_x_0+C_x_u*(C_z_q+2*mu_c))+\
            C_m_q*(C_x_u*(C_z_a_dot-2*mu_c)-2*mu_c*C_z_a)+\
            2*mu_c*Kyy2*(C_x_a*C_z_u-C_z_a*C_x_u)
        D = C_m_u*(C_x_a*(C_z_q+2*mu_c)-C_z_0*(C_z_a_dot-2*mu_c))-\
            C_m_a*(2*mu_c*C_x_0 + C_x_u*(C_z_q+2*mu_c))+\
            C_m_a_dot*(C_x_0*C_x_u-C_z_0*C_z_u)+C_m_q*(C_x_u*C_z_a-C_z_u*C_x_a)
        E = -C_m_u*(C_x_0*C_x_a+C_z_0*C_z_a)+C_m_a*(C_x_0*C_x_u+C_z_0*C_z_u)
        R = B*C*D-A*D**2-B**2*E
        if A<0 and R<0:
            print("The solution is stable for the symmetric case")
        if A>0 and R>0:
            print("The solution is stable for the symmetric case")
        if (A>0 and R<0) or (A<0 and R>0):
            print("The solution is unstable for the symmetric case!")
        return

    def get_eigenvalues(self, Sys,V,sit):

        """
        Input:
        Sys: State-Space System
        :param V: Initial Airspeed
        :param sit: "sym" or "asym"
        :return: eigenvalues
        """

        if sit =="sym":
            const = V/self.c
        if sit =="asym":
            const = V/self.b
        return np.linalg.eigvals(Sys.A)*const**(-1)

    def get_period_prop(self,Sys,V,sit):
        """
        Inputs:
        :param Sys: State-Space System
        :param V: Initial Airspeed
        :param sit: "sym"
        :return: [lambda1: T_1/2, period, damping,lambda2:...]
        """
        eig = self.get_eigenvalues(Sys,V,sit)
        prop = []
        ii = [1,3]
        for i in ii:
            li = eig[i]
            Ti = abs(np.log(0.5)/li.real)*self.c/V
            Pi = abs(2*np.pi/li.imag)*self.c/V
            zetai = abs(li.real/(sqrt(li.real**2+li.imag**2)))
            prop.append(["For \u03BB_%.0f,%.0f:  T_1/2 = %.4f [s]; P = %.4f [s]; damping = %.4f [-]"
                         %(i,i+1,Ti,Pi,zetai)])
        return prop
    def get_period_prop_a(self,Sys,V,sit):
        """
        Inputs:
        :param Sys: State-Space System
        :param V: Initial Airspeed
        :param sit: "asym"
        :return: [lambda1: T_1/2, period, damping,lambda2:...]
        """
        eig = self.get_eigenvalues(Sys,V,sit)
        prop = []
        ii = [1,2,3,4]
        for i in ii:
            li = eig[i-1]
            Ti = abs(np.log(0.5)/li.real)*self.b/V
            Pi = abs(2*np.pi/li.imag)*self.b/V
            zetai = abs(li.real/(sqrt(li.real**2+li.imag**2)))
            prop.append(["For \u03BB_%.0f :  T_1/2 = %.3f [s]; P = %.3f [s]; damping = %.3f [-]"%(i,Ti,Pi,zetai)])
        return prop

    def compute_asym_sys(self,V,Kxx2,Kzz2,Kxz, C_L,C_Y,C_l,C_n):
        """
        Inputs:
        :param V: Airspeed
        :param Kxx2: Normalised moment of inertia around x-axis ^2
        :param Kzz2: Normalised moment of inertia around z-axis ^2
        :param Kxz: Normalised cross moment of inertia xz
        :param C_L: Lift coefficient
        :param C_Y: array containing all stability derivatives and control derivatives related to C_Y
        :param C_l: array containing all stability derivatives and control derivatives to C_l
        :param C_n: array containing all stability derivatives and control derivatives to C_n
        :return: Asymmetric state-space system
        """
        mu_b = self.get_mub(self.W,self.h,self.S,self.b)
        C_y_b, C_y_b_dot, C_y_p, C_y_r, C_y_da, C_y_dr = C_Y
        C_l_b, C_l_p, C_l_r,  C_l_da, C_l_dr = C_l
        C_n_b, C_n_b_dot, C_n_p, C_n_r, C_n_da,C_n_dr = C_n
        C_1 = np.array([[(C_y_b_dot-2 * mu_b )* self.b / V, 0, 0, 0],
                        [0, -0.5 * self.b / V, 0, 0],
                        [0, 0, -4*mu_b*Kxx2*self.b/V, 4*mu_b*Kxz*self.b/V],
                        [C_n_b_dot*self.b/V, 0, 4*mu_b*Kxz*self.b/V, -4 * mu_b * Kzz2 * self.b / V]])
        C_2 = np.array([[C_y_b, C_L, C_y_p, (C_y_r-4*mu_b)],
                        [0, 0, 1, 0],
                        [C_l_b, 0, C_l_p, C_l_r],
                        [C_n_b, 0, C_n_p, C_n_r]])
        C_3 = np.array([[C_y_da,C_y_dr],
                        [0,0],
                        [C_l_da,C_l_dr],
                        [C_n_da,C_n_dr]])
        A_s = -np.dot(np.linalg.inv(C_1), C_2)
        B_s = -np.dot(np.linalg.inv(C_1), C_3)
        C_s = np.identity(4)
        D_s = np.array([[0,0],
                        [0,0],
                        [0,0],
                        [0,0]])
        return matlab.ss(A_s, B_s, C_s, D_s)

    def dimensionalise_sym(self, x, V):
        return np.array([x[0]*V, x[1], x[2], x[3] * V / self.c])

    def phugoid(self, T, delta_e, V,sys):
        alpha_init = self.alpha0
        theta_init = self.th0
        q_init = self.q0
        #u = matlab.impulse(self.sym_sys, T,output=0)* delta_e
        #alpha = matlab.impulse(self.sym_sys, T,output=1)* delta_e
        #theta = matlab.impulse(self.sym_sys, T,output=2)* delta_e
        #q = matlab.impulse(self.sym_sys, T,output=3)* delta_e
        init = np.array([0, alpha_init, theta_init, q_init * self.c / V])
        y, T, x = matlab.lsim(sys, U=delta_e, T=T, X0=init)
        return self.dimensionalise_sym(y.transpose(), V)

    def short_period(self, T, delta_e, V,sys):
        alpha_init = self.alpha0
        theta_init = self.th0
        q_init = self.q0
        #u = matlab.step(self.sym_sys, T, output=0) * delta_e
        #alpha = matlab.step(self.sym_sys, T, output=1) * delta_e
        #theta = matlab.step(self.sym_sys, T, output=2) * delta_e
        #q = matlab.step(self.sym_sys, T, output=3) * delta_e
        init = np.array([0, alpha_init, theta_init, q_init * self.c / V])
        y, T, x = matlab.lsim(sys,U=delta_e,T=T, X0=init)
        return self.dimensionalise_sym(y.transpose(), V)

    def dimensionalise_asym(self, x, V):
        return np.array([x[0], x[1], x[2] * 2*V / self.b, x[3] * 2*V / self.b])

    def dutch_roll(self, T, delta_r, V, delta_a=None):
        beta_init = self.beta0
        phi_init = self.phi0
        p_init = self.p0
        r_init = self.r0
        zero = np.zeros(len(T))
        if delta_a is None:
            delta =np.vstack((zero,delta_r))
        else:
            delta = np.vstack((delta_a, delta_r))
        delta = delta.transpose()
        #print(delta_r)
        #beta = matlab.impulse(self.asym_sys, T, input = 1, output = 0)*delta_r
        #phi = matlab.impulse(self.asym_sys, T, input = 1, output = 1)*delta_r
        #p = matlab.impulse(self.asym_sys, T, input = 1, output = 2)*delta_r
        #r = matlab.impulse(self.asym_sys, T, input = 1, output = 3)*delta_r
        init = np.array([beta_init, phi_init, p_init * self.b / (2 * V), r_init * self.b / (2 * V)])
        y,T,x = matlab.lsim(self.asym_sys,U=delta,T=T, X0=init)
        beta, phi, p, r = self.dimensionalise_asym(y.transpose(), V)
        # TODO: do we need to specify a theta_init here?
        return beta, phi, p, r, self.psi(T, r)

    def ap_roll(self, T, delta_a, V, delta_r=None):
        beta_init = self.beta0
        phi_init = self.phi0
        p_init = self.p0
        r_init = self.r0
        zero = np.zeros(len(T))
        if delta_r is None:
            delta =np.vstack((delta_a,zero))
        else:
            delta = np.vstack((delta_a, delta_r))
        delta = delta.transpose()
        #print("U_delta_a = ", delta_a)
        #beta = matlab.step(self.asym_sys, T, input=0, output=0) * delta_a
        #phi = matlab.step(self.asym_sys, T, input=0, output=1) * delta_a
        #p = matlab.step(self.asym_sys, T, input=0, output=2) * delta_a
        #r = matlab.step(self.asym_sys, T, input=0, output=3) * delta_a
        init = np.array([beta_init, phi_init, p_init * self.b / (2 * V), r_init * self.b / (2 * V)])
        y, T, x = matlab.lsim(self.asym_sys, U=delta, T=T, X0=init)
        beta, phi, p, r = self.dimensionalise_asym(y.transpose(), V)
        # TODO: do we need to specify a theta_init here?
        return beta, phi, p, r, self.psi(T, r)

    def spiral(self, T, delta_a, V, delta_r=None):
        beta_init = self.beta0
        phi_init = self.phi0
        p_init = self.p0
        r_init = self.r0
        zero = np.zeros(len(T))
        if delta_r is None:
            delta = np.vstack((delta_a, zero))
        else:
            delta = np.vstack((delta_a, delta_r))
        delta = delta.transpose()
        #print(delta_a)
        #beta = matlab.impulse(self.asym_sys, T, input=0, output=0) * delta_a
        #phi = matlab.impulse(self.asym_sys, T, input=0, output=1) * delta_a
        #p = matlab.impulse(self.asym_sys, T, input=0, output=2) * delta_a
        #r = matlab.impulse(self.asym_sys, T, input=0, output=3) * delta_a
        init = np.array([beta_init, phi_init, p_init * self.b / (2 * V), r_init * self.b / (2 * V)])
        y, T, x = matlab.lsim(self.asym_sys, U=delta, T=T, X0=init)
        beta, phi, p, r = self.dimensionalise_asym(y.transpose(), V)
        # TODO: do we need to specify a theta_init here?
        return beta, phi, p, r, self.psi(T, r)

    def psi(self,T, r):
        theta_init = self.th0
        return sp.cumtrapz(r / np.cos(theta_init), T, initial=0)

    def chi(self,T, beta, r):
        theta_init = self.th0
        beta = np.delete(beta, 0)
        return beta + self.psi(T, r, theta_init)
