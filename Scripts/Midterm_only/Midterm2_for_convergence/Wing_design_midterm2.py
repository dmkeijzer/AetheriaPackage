import numpy as np
from Airfoil_analysis_midterm2 import airfoil_stats, airfoil_datapoint
from scipy.interpolate import interp1d

def deps_da(Lambda_quarter_chord, b,lh, h_ht, A, CLaw):
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
    mtv = h_ht * 2 / b
    Keps = (0.1124 + 0.1265 * Lambda_quarter_chord + 0.1766 * Lambda_quarter_chord ** 2) / r ** 2 + 0.1024 / r + 2
    Keps0 = 0.1124 / r ** 2 + 0.1024 / r + 2
    v = 1 + (r ** 2 / (r ** 2 + 0.7915 + 5.0734 * mtv ** 2) ** (0.3113))
    de_da = Keps / Keps0 * CLaw / (np.pi * A) * (
            r / (r ** 2 + mtv ** 2) * 0.4876 / (np.sqrt(r ** 2 + 0.6319 + mtv ** 2)) + v * (
            1 - np.sqrt(mtv ** 2 / (1 + mtv ** 2))))
    # print("Configuration %.0f de/da = %.4f "%(conf,de_da))
    return de_da


def winglet_dAR(AR, h_wl, b):  # Gundmundsson 10.5 Wingtip design

    return 1.9*(h_wl/b)*AR


class wing_design:

    def __init__(self, AR, s1, sweepc41, s2, sweepc42, M, S, lh, h_ht, w, h_wl1, h_wl2):
        self.AR_b = AR
        self.s1 = s1
        self.S1 = s1 * S
        self.sweepc41 = sweepc41
        self.s2 = s2
        self.S2 = s2 *S
        self.sweepc42= sweepc42
        airfoil = airfoil_stats()
        self.clmax = airfoil[0]
        self.Cl_Cdmin = airfoil[2]
        self.Clalpha = airfoil[4] * 180/np.pi
        self.a_0L = airfoil[8]
        self.a_saf = airfoil[7]
        self.M = M
        self.S = S
        self.lh = lh
        self.h_ht = h_ht
        self.w = w
        self.h_wl1 = h_wl1
        self.h_wl2 = h_wl2

    def taper_opt(self):
        return 0.45 * np.exp(-0.036 * self.sweepc41), 0.45 * np.exp(-0.036 * self.sweepc42)  # Eq. 7.4 Conceptual Design of a Medium Range Box Wing Aircraft

    def wing_planform_double(self):
        # Wing 1
        self.taper1 = self.taper_opt()[0]
        self.taper2 = self.taper_opt()[1]
        b1 = np.sqrt(2 * self.AR_b * self.S1)
        c_r1 = 2 * self.S1 / ((1 + self.taper1) * b1)
        c_t1 = self.taper1 * c_r1
        c_MAC1 = (2 / 3) * c_r1 * ((1 + self.taper1 + self.taper1 ** 2) / (1 + self.taper1))
        y_MAC1 = (b1 / 6) * ((1 + 2 * self.taper1) / (1 + self.taper1))
        tan_sweep_LE1 = 0.25 * (2 * c_r1 / b1) * (1 - self.taper1) + np.tan(self.sweepc41)

        X_LEMAC1 = y_MAC1 * tan_sweep_LE1
        wing1 = [b1, c_r1, c_t1, c_MAC1, y_MAC1, X_LEMAC1]

        # Wing 2

        b2 = np.sqrt(2 * self.AR_b * self.S2)
        c_r2 = 2 * self.S2 / ((1 + self.taper2) * b2)
        c_t2 = self.taper2 * c_r2
        c_MAC2 = (2 / 3) * c_r2 * ((1 + self.taper2 + self.taper2 ** 2) / (1 + self.taper2))
        y_MAC2 = (b2 / 6) * ((1 + 2 * self.taper2) / (1 + self.taper2))
        tan_sweep_LE2 = 0.25 * (2 * c_r2 / b2) * (1 - self.taper2) + np.tan(self.sweepc42)

        X_LEMAC2 = y_MAC2 * tan_sweep_LE2
        wing2 = [b2, c_r2, c_t2, c_MAC2, y_MAC2, X_LEMAC2]

        return wing1, wing2

    def sweep_atx(self, x):
        wg = self.wing_planform_double()
        b1 = wg[0][0]
        c_r1 =  wg[0][1]
        b2 = wg[1][0]
        c_r2 = wg[1][1]
        tan_sweep_LE1 = 0.25 * (2 * c_r1 / b1) * (1 - self.taper1) + np.tan(self.sweepc41)
        sweep1 = np.arctan(tan_sweep_LE1- x * (2 * c_r1 / b1) * (1 - self.taper1))
        tan_sweep_LE2 = 0.25 * (2 * c_r2 / b2) * (1 - self.taper2) + np.tan(self.sweepc42)
        sweep2 = np.arctan(tan_sweep_LE2 - x * (2 * c_r2 / b2) * (1 - self.taper2))
        return sweep1, sweep2

    def liftslope(self):
        beta = np.sqrt(1 - self.M ** 2)
        SW = np.tan(self.sweep_atx(0.5))
        wg = self.wing_planform_double()

        self.AR_i = 2 * self.AR_b +  winglet_dAR(2 * self.AR_b, self.h_wl1, wg[0][0])
        slope1 = self.Clalpha * (self.AR_i / (2 + np.sqrt(4 + ((self.AR_i * beta / 0.95) ** 2) * ((1 + SW ** 2) / (beta ** 2)))))
        self.AR_2 = 2 * self.AR_b + winglet_dAR(2 * self.AR_b, self.h_wl2, wg[0][0])
        slope2_b = self.Clalpha * (self.AR_2 / (2 + np.sqrt(4 + ((self.AR_2 * beta / 0.95) ** 2) * ((1 + SW ** 2) / (beta ** 2)))))
        deda = deps_da(self.sweepc41, wg[0][0], self.lh, self.h_ht, self.AR_i, slope1)
        slope2 = slope2_b * (1 - deda)

        wfi = 1 + 0.025*(self.w/wg[0][0]) - 0.25*(self.w/wg[0][0])  # wing fuselage interaction factor: effect of fuselage diameter on aerodynamic characteristics for straightwing at low and high aspect ratio

        slope_tot = wfi*(slope1 * self.s1 + slope2 * self.s2)
        return slope_tot, slope1*wfi, slope2*wfi, deda, slope1, slope2

    def CLmax_s(self):
        ls = self.liftslope()
        CLa = ls[0]
        deda = ls[3]
        CLmax1 = self.clmax * 0.9

        alpha_s2 = round(((self.a_saf-self.a_0L)*(1-deda[1]) + self.a_0L)*4)/4
        CLmax2 = 0.9 * airfoil_datapoint("CL", "Stall", alpha_s2)
        CLmax = self.s1*CLmax1 + self.s2*CLmax2
        self.a_s = (180/np.pi) * CLmax/CLa + self.a_0L
        return CLmax, CLmax1, CLmax2, self.a_s

    def post_stall_lift_drag(self, tc, CDs_W, CDs_f, Afus):
        # Wing
        stall = self.CLmax_s()
        CLs = stall[0]
        a_s = self.a_s[1] * np.pi/180
        A1 = 0.5*(1.1 + 0.018 * self.AR_i)
        A2 = (CLs - 2*A1*np.sin(a_s)*np.cos(a_s))*(np.sin(a_s)/(np.cos(a_s)**2))
        CDmax = (1 + 0.065*self.AR_i)/(0.9 + tc)
        B2 = (CDs_W - CDmax * np.sin(a_s))/np.cos(a_s)

        alpha_ps = (np.pi/180)*np.arange(round(self.a_s[1])+1, 91, 1)
        CL_ps = A1*np.sin(2*alpha_ps)+A2*((np.cos(alpha_ps)**2)/np.sin(alpha_ps))
        CD_ps = CDmax*np.sin(alpha_ps) + B2 * np.cos(alpha_ps)

        # Fuselage
        CDmax_f = 1.18*Afus/self.S  # https://sv.20file.org/up1/916_0.pdf Drag coefficient of a cylinder
        B2_f = (CDs_f - CDmax_f * np.sin(a_s)) / np.cos(a_s)
        CD_ps_f = CDmax_f * np.sin(alpha_ps) + B2_f * np.cos(alpha_ps)
        return alpha_ps, CL_ps, CD_ps, CD_ps_f

    def CLa(self, tc, CDs_W, CDs_f, Afus, alpha_lst):

        poststall = self.post_stall_lift_drag(tc, CDs_W, CDs_f, Afus)
        CLa = self.liftslope()
        alpha = np.arange(-5,self.a_s[1],0.25)
        CL = (np.pi/180)*CLa[0][0]*(alpha - self.a_0L)
        alpha = np.append(alpha,poststall[0]*180/np.pi)
        CL = np.append(CL,poststall[1])

        curve = interp1d(alpha,CL, kind = 'quadratic')
        return curve(alpha_lst)

    def CDa_poststall(self, tc, CDs_W, CDs_f, Afus, alpha_lst, type, CD):

        drag_post = self.post_stall_lift_drag(tc, CDs_W, CDs_f, Afus)
        if type == "wing":
            alpha_pre = np.arange(0, self.a_s[1], 0.1)
            CL_lst = self.CLa(tc, CDs_W, CDs_f, Afus, alpha_pre)
            CD_pre_lst = CD(CL_lst) - CDs_f
            alpha = np.append(alpha_pre, drag_post[0]*180/np.pi)
            drag_w = np.append(CD_pre_lst,drag_post[2])
            fdrag = interp1d(alpha, drag_w)
            return fdrag(alpha_lst)

        elif type == "fus":
            newal = np.arange(0, self.a_s[1] -1, 1)
            alpha = np.append(newal, drag_post[0]*180/np.pi)
            drag_f = np.append(CDs_f*np.ones(len(newal)), drag_post[3])
            # print(alpha_lst)
            # print(newal, drag_f)
            fdrag = interp1d(alpha, drag_f)
            return fdrag(alpha_lst)

