import numpy as np
from Airfoil_analysis import airfoil_stats, airfoil_datapoint
from scipy.interpolate import interp1d
from scipy.stats import linregress
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
    #print("Configuration %.0f de/da = %.4f "%(conf,de_da))
    #print(de_da)
    return de_da*0.5

def winglet_dAR(AR, h_wl, b): # Gundmundsson 10.5 Wingtip design

    return 1.9*(h_wl/b)*AR

def winglet_factor(h_wl, b, k_wl):  #https://www.fzt.haw-hamburg.de/pers/Scholz/Aero/AERO_PUB_Winglets_IntrinsicEfficiency_CEAS2017.pdf

    return (1+(2/k_wl)*(h_wl/b))**2


class wing_design:

    def __init__(self, AR1, AR2, s1, sweepc41, s2, sweepc42, M, S, lh, h_ht, w, h_wl1,h_wl2, k_wl, i1):
        self.AR1 = AR1
        self.AR2 = AR2
        self.s1 = s1
        self.S1 = s1 * S
        self.sweepc41 = sweepc41
        self.s2 = s2
        self.S2 = s2 *S
        self.sweepc42= sweepc42
        airfoil = airfoil_stats()
        self.clmax = airfoil[0]
        self.Cl_Cdmin = airfoil[2]
        self.Clalpha = airfoil[4]* 180/np.pi
        self.a_0L = airfoil[8]
        self.a_saf = airfoil[7]
        self.M = M
        self.S = S
        self.lh = lh
        self.h_ht = h_ht
        self.w = w
        self.h_wl1 = h_wl1
        self.h_wl2 = h_wl2
        self.k_wl = k_wl
        self.i1 = i1  # trim angle for first wing
    def taper_opt(self):
        return 0.45 * np.exp(-0.036 * self.sweepc41), 0.45 * np.exp(-0.036 * self.sweepc42)  # Eq. 7.4 Conceptual Design of a Medium Range Box Wing Aircraft

    def wing_planform_double(self):
        # Wing 1
        self.taper1 = self.taper_opt()[0]
        self.taper2 = self.taper_opt()[1]
        b1 = np.sqrt( self.AR1 * self.S1)
        c_r1 = 2 * self.S1 / ((1 + self.taper1) * b1)
        c_t1 = self.taper1 * c_r1
        c_MAC1 = (2 / 3) * c_r1 * ((1 + self.taper1 + self.taper1 ** 2) / (1 + self.taper1))
        y_MAC1 = (b1 / 6) * ((1 + 2 * self.taper1) / (1 + self.taper1))
        tan_sweep_LE1 = 0.25 * (2 * c_r1 / b1) * (1 - self.taper1) + np.tan(self.sweepc41)

        X_LEMAC1 = y_MAC1 * tan_sweep_LE1
        AReff1 = self.AR1 * winglet_factor(self.h_wl1, b1, self.k_wl) #+winglet_dAR(self.AR1, self.h_wl1, b1)
        beff1 = b1*np.sqrt(AReff1/self.AR1)
        wing1 = [b1, c_r1, c_t1, c_MAC1, y_MAC1, X_LEMAC1, AReff1, beff1]
        # Wing 2

        b2 = np.sqrt( self.AR2 * self.S2)
        c_r2 = 2 * self.S2 / ((1 + self.taper2) * b2)
        c_t2 = self.taper2 * c_r2
        c_MAC2 = (2 / 3) * c_r2 * ((1 + self.taper2 + self.taper2 ** 2) / (1 + self.taper2))
        y_MAC2 = (b2 / 6) * ((1 + 2 * self.taper2) / (1 + self.taper2))
        tan_sweep_LE2 = 0.25 * (2 * c_r2 / b2) * (1 - self.taper2) + np.tan(self.sweepc42)

        X_LEMAC2 = y_MAC2 * tan_sweep_LE2
        AReff2 = self.AR2 * winglet_factor(self.h_wl2, b2, self.k_wl) #+ winglet_dAR(self.AR2, self.h_wl2, b2)
        beff2 = b1 * np.sqrt(AReff2 / self.AR2)
        wing2 = [b2, c_r2, c_t2, c_MAC2, y_MAC2, X_LEMAC2, AReff2, beff2]
        #print(wing2)

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
    def liftslope(self, deda):
        beta = np.sqrt(1 - self.M ** 2)
        SW = np.tan(self.sweep_atx(0.5))
        wg = self.wing_planform_double()

        self.AR_i = self.AR1 +  winglet_dAR(self.AR2, self.h_wl1, wg[0][0])
        slope1 = self.Clalpha * (self.AR_i / (2 + np.sqrt(4 + ((self.AR_i * beta / 0.95) ** 2) * ((1 + SW ** 2) / (beta ** 2)))))
        self.AR_2 = self.AR2 + winglet_dAR(self.AR2, self.h_wl2, wg[0][0])
        slope2_b = self.Clalpha * (self.AR_2 / (2 + np.sqrt(4 + ((self.AR_2 * beta / 0.95) ** 2) * ((1 + SW ** 2) / (beta ** 2)))))
        slope2 = slope2_b * (1 - deda)
        # print("de_da",deda)
        wfi = 1 + 0.025*(self.w/wg[0][0]) - 0.25*(self.w/wg[0][0])  # wing fuselage interaction factor: effect of fuselage diameter on aerodynamic characteristics for straightwing at low and high aspect ratio

        slope_tot = wfi*(slope1 * self.s1 + slope2 * self.s2)
        # print("Check:", slope_tot, slope1*wfi, slope2_b*wfi, deda, slope2*wfi, slope1, slope2)
        return slope_tot, slope1*wfi, slope2_b*wfi, deda, slope2*wfi, slope1, slope2

    def CLmax_s(self, de_da):
        ls = self.liftslope(de_da)
        CLa = ls[0]
        deda = ls[3]
        CLmax1 = self.clmax *0.9

        alpha_s2 = np.round(((self.a_saf-self.i1 -self.a_0L)*(1-deda) +self.a_0L)*4)/4
        CLmax2 = 0.9* airfoil_datapoint("CL", "Stall",alpha_s2)
        CLmax = self.s1*CLmax1 +self.s2*CLmax2
        self.a_s = (180/np.pi)* CLmax/CLa + self.a_0L
        return CLmax, CLmax1, CLmax2, self.a_s

    def post_stall_lift_drag(self, tc, CDs_W, CDs_f, Afus, de_da):
        #Wing
        stall = self.CLmax_s(de_da)
        CLs = stall[0]
        a_s = self.a_s[1]* np.pi/180
        A1 = 0.5*(1.1 + 0.018* self.AR_i)
        A2 = (CLs - 2*A1*np.sin(a_s)*np.cos(a_s))*(np.sin(a_s)/(np.cos(a_s)**2))
        CDmax = (1 + 0.065*self.AR_i)/(0.9 + tc)
        B2 = (CDs_W - CDmax * np.sin(a_s))/np.cos(a_s)

        alpha_ps = (np.pi/180)*np.arange(np.round(self.a_s[1])+1, 91, 1)
        CL_ps = A1*np.sin(2*alpha_ps)+A2*((np.cos(alpha_ps)**2)/np.sin(alpha_ps))
        CD_ps = CDmax*np.sin(alpha_ps)+ B2 * np.cos(alpha_ps)

        # Fuselage
        CDmax_f = 1.18*Afus/self.S # https://sv.20file.org/up1/916_0.pdf Drag coefficient of a cylinder
        B2_f = (CDs_f - CDmax_f * np.sin(a_s)) / np.cos(a_s)
        CD_ps_f = CDmax_f * np.sin(alpha_ps) + B2_f * np.cos(alpha_ps)
        return alpha_ps, CL_ps, CD_ps, CD_ps_f

    def CLa(self, tc, CDs_W, CDs_f, Afus, alpha_lst, de_da):

        poststall = self.post_stall_lift_drag(tc, CDs_W, CDs_f, Afus, de_da)
        CLa = self.liftslope(de_da)
        alpha_i = np.arange(-5,self.a_s[1],0.25)
        CL1 = (np.pi/180)*CLa[1][0]*(alpha_i +self.i1- self.a_0L)
        CL2 = (np.pi/180)*CLa[4][0]*(alpha_i - self.a_0L)
        CL = self.s1*CL1 + self.s2*CL2
        alpha = np.append(alpha_i,poststall[0]*180/np.pi)
        CL = np.append(CL,poststall[1])
        CL1i = (np.pi / 180) * CLa[1][0] * (alpha + self.i1 - self.a_0L)
        CL2i = (np.pi / 180) * CLa[4][0] * (alpha - self.a_0L)

        curve_tot = interp1d(alpha,CL, kind = 'quadratic')
        curve1 = interp1d(alpha,CL1i)
        curve2 = interp1d(alpha, CL2i)
        return curve_tot(alpha_lst), curve1(alpha_lst), curve2(alpha_lst)

    def CDa_poststall(self, tc, CDs_W, CDs_f, Afus, alpha_lst, type, CD, de_da):

        drag_post = self.post_stall_lift_drag(tc, CDs_W, CDs_f, Afus, de_da)
        if type=="wing":
            alpha_pre= np.arange(-3,self.a_s[1], 0.1)
            CL_lst = self.CLa(tc, CDs_W, CDs_f, Afus, alpha_pre, de_da)[0]
            CD_pre_lst = CD(CL_lst)- CDs_f
            alpha = np.append(alpha_pre, drag_post[0]*180/np.pi)
            drag_w = np.append(CD_pre_lst,drag_post[2])
            fdrag = interp1d(alpha, drag_w)
            return fdrag(alpha_lst)
        elif type == "fus":
            newal = np.arange(-3,self.a_s[1] -1,1)
            alpha = np.append(newal,drag_post[0]*180/np.pi)
            drag_f = np.append(CDs_f*np.ones(len(newal)), drag_post[3])

            fdrag = interp1d(alpha, drag_f)
            return fdrag(alpha_lst)
    def C_T(self,ne1,ne2, T, V_inf, rho):
        self.T = T
        self.V_inf = V_inf
        self.rho = rho
        return (ne1+ne2)*self.T/(0.5*self.rho*((V_inf)**2)*self.S)

    def CL_T(self,T, V_inf, rho,  alpha_wp, ne1,ne2):
        self.alphawp = alpha_wp
        return np.sin(self.alphawp*(np.pi/180))*self.C_T(ne1,ne2, T, V_inf, rho)
    def deltaV(self, T, V_inf, rho, D, ne1 , ne2):
        self.D = D
        return V_inf * ( np.sqrt(1+ (self.C_T(ne1,ne2, T, V_inf, rho)*self.S)/((ne1+ne2)*(np.pi/4)*self.D**2))-1)
    def Deff(self, T, V_inf, rho, D , ne1 , ne2):
        return D*np.sqrt((V_inf+self.deltaV(T, V_inf, rho, D, ne1 , ne2)*0.5)/(V_inf+self.deltaV(T, V_inf, rho, D , ne1, ne2)))
    def a_ss(self, T, V_inf, rho, D ,alpha_wp, ne1, ne2):

        return (180/np.pi)*np.arctan2(V_inf*np.sin(alpha_wp*np.pi/180),(V_inf*np.cos(alpha_wp*np.pi/180)+ self.deltaV(T, V_inf, rho, D , ne1, ne2)*0.5)) - self.a_0L
    def Aseff(self, n_e1, n_e2, T, V_inf, rho, D):
        self.n_e1 = n_e1
        self.n_e2 = n_e2
        self.b1 = np.sqrt(self.AR1*self.S1)
        self.b2 = np.sqrt(self.AR2*self.S2)
        AS1 = (self.n_e1*D**2)/self.b1
        AS2 = (self.n_e2*D**2)/self.b2
        ASeff1 = AS1+ (self.AR1 - AS1)*(V_inf/(self.deltaV( T, V_inf, rho, D , n_e1, n_e2) + V_inf))**(self.AR1-AS1)
        ASeff2 = AS2 + (self.AR2 - AS2) * (V_inf / (self.deltaV( T, V_inf, rho, D, n_e1, n_e2) + V_inf)) ** (self.AR2 - AS2)
        return ASeff1, ASeff2
    def CL_W_S(self,  T, V_inf, rho, D , n_e1, n_e2, tc, CDs_W, CDs_f, Afus, alpha_wp, de_da):
        self.b1 = np.sqrt(self.AR1 * self.S1)
        self.b2 = np.sqrt(self.AR2 * self.S2)
        SW = np.tan(self.sweep_atx(0.5))
        beta = np.sqrt(1 - self.M ** 2)
        ARS1 = self.Aseff(n_e1, n_e2, T, V_inf, rho, D)[0]
        ARS2 = self.Aseff(n_e1, n_e2, T, V_inf, rho, D)[1]
        slope1p = self.Clalpha * (ARS1 / (2 + np.sqrt(4 + ((ARS1 * beta / 0.95) ** 2) * ((1 + SW ** 2) / (beta ** 2)))))
        slope2p = self.Clalpha * (ARS2 / (2 + np.sqrt(4 + ((ARS2 * beta / 0.95) ** 2) * ((1 + SW ** 2) / (beta ** 2)))))*(1-de_da)

        pt1 = 2*self.CLa(tc, CDs_W, CDs_f, Afus, alpha_wp, de_da)[1]/(np.pi*self.AR1)
        pt2 = 2*self.CLa(tc, CDs_W, CDs_f, Afus, alpha_wp, de_da)[2]/(np.pi*self.AR2)
        pt3 = 2*slope1p[0]*np.sin(self.a_ss(T, V_inf, rho, D , alpha_wp, n_e1, n_e2)*np.pi/180)/(np.pi*ARS1)#2*self.CLa(tc, CDs_W, CDs_f, Afus, self.a_ss(T, V_inf, rho, D , alpha_wp, n_e1, n_e2), de_da)[1]*np.sin(self.a_ss(T, V_inf, rho, D , alpha_wp, n_e1, n_e2)*np.pi/180)/(np.pi*ARS1)    #self.a_ss(T, V_inf, rho, D , alpha_wp)
        pt4 = 2*slope2p[1]*np.sin(self.a_ss(T, V_inf, rho, D , alpha_wp, n_e1, n_e2)*np.pi/180)/(np.pi*ARS2)  #*np.sin(self.a_ss(T, V_inf, rho, D , alpha_wp, n_e1, n_e2)*np.pi/180)
        self.CLW1 = (2/self.S1)*((np.pi/4)*self.b1**2 - n_e1*(np.pi/4)*(self.Deff(T, V_inf, rho, D, n_e1, n_e2))**2)*pt1
        self.CLW2 = (2 / self.S2) * ((np.pi / 4) * self.b2 ** 2 - n_e2*(np.pi / 4) *(  self.Deff(T, V_inf, rho, D, n_e1, n_e2))**2)*pt2
        self.CLS1 = n_e1*(np.pi*(self.Deff(T, V_inf, rho, D, n_e1, n_e2)**2 )/(2*self.S1))*((V_inf +self.deltaV( T, V_inf, rho, D, n_e1, n_e2))**2/V_inf**2)* pt3
        self.CLS2 = n_e2 * (np.pi * (self.Deff(T, V_inf, rho, D, n_e1, n_e2)**2) / (2 * self.S2)) * ((V_inf + self.deltaV( T, V_inf, rho, D, n_e1, n_e2)) ** 2 / V_inf ** 2)*pt4
        CLWS1 = self.CLW1 + self.CLS1
        CLWS2 = self.CLW2 + self.CLS2
        return CLWS1 , CLWS2

    # T_per_eng_during_stall, V_stall, rho, prop_radius * 2, n_prop_1,
    # n_prop_2, const.tc, CDs_w, CDs_f, Afus, alpha_wp, de_da

    def CLa_wprop(self, T, V_inf, rho, D, ne1,ne2, tc, CDs_W, CDs_f, Afus, alpha_wp, de_da):
        self.CLmax_s(de_da)

        alpha = np.arange(-5, self.a_s[1]+1, 0.25)
        CLaw1 = self.CL_W_S(  T, V_inf, rho, D ,ne1,ne2, tc, CDs_W, CDs_f, Afus, alpha, de_da)[0]+ self.CL_T(T, V_inf, rho,  alpha, ne1,ne2)
        CLaw2 = self.CL_W_S(  T, V_inf, rho, D , ne1,ne2,tc, CDs_W, CDs_f, Afus, alpha, de_da)[1] + self.CL_T(T, V_inf, rho,  alpha, ne1,ne2)
        CLaw = self.s1 *CLaw1 + self.s2*CLaw2
        slope1wp = linregress((np.pi/180)*alpha[0:int(len(alpha))], CLaw1[0:int(len(alpha))])[0]
        slope2wp = linregress((np.pi/180)*alpha[0:int(len(alpha))], CLaw2[0:int(len(alpha))])[0]*(1/(1-de_da))

        curve = interp1d(alpha, CLaw, kind='quadratic')
        curve1 = interp1d(alpha, CLaw1)
        curve2 = interp1d(alpha, CLaw2)
        CLmaxwp = curve(self.a_s[1])
        CLmaxwp1 = curve1(self.a_s[1])
        CLmaxwp2 = curve2(self.a_s[1])
        return curve(alpha_wp), min(CLmaxwp,3), slope1wp, slope2wp, min(CLmaxwp1,3), min(CLmaxwp2,3), curve1(alpha_wp), curve2(alpha_wp)



