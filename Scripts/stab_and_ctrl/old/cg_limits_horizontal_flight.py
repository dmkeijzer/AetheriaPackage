import numpy as np
from scipy.linalg import null_space
from matplotlib import pyplot as plt
from matplotlib import colors as mc
import colorsys
from dataclasses import dataclass
from itertools import combinations
import os
import json

root_path = os.path.join(os.getcwd(), os.pardir)

taper_ratio = 0.4
a = 340
def Speeds(conf):
    a = 340
    if conf==1:
        datafile = open(os.path.join(root_path, "data/inputs_config_1.json"), "r")
        data = json.load(datafile)
        datafile.close()
        Afwd = data["Aerodynamics"]["AR"] * 2
    if conf==2:
        datafile = open(os.path.join(root_path, "data/inputs_config_2.json"), "r")
        data = json.load(datafile)
        datafile.close()
        Afwd = data["Aerodynamics"]["AR"] * 2
    if conf==3:
        datafile = open(os.path.join(root_path, "data/inputs_config_3.json"), "r")
        data = json.load(datafile)
        datafile.close()
        Afwd = data["Aerodynamics"]["AR"]
    V_c = data["Flight performance"]["V_cruise"]
    V_s = data["Requirements"]["V_stall"]
    M_s = V_s/a
    M_c = V_c/a
    beta_s = np.sqrt(1 - M_s ** 2)
    beta_c = np.sqrt(1 - M_c ** 2)
    print("Configuration %.0f at STALL beta*AR = %.3f"%(conf,beta_s * Afwd))
    print("Configuration %.0f at CRUISE beta*AR = %.3f "%(conf,beta_c * Afwd))
    return V_c, V_s, M_c,M_s

# V1 = Speeds(1)
# V2 = Speeds(2)
# V3 = Speeds(3)

def deps_da(Lambda_quarter_chord, b,lh, h_ht, A, CLaw,conf):
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
    return de_da

def lh(xacfwd,xacrear):
    return abs(xacfwd-xacrear)

def C_L_a(conf, cruise,A, Lambda_half, eta=0.95):
    """
    Inputs:
    :param M: Mach number
    :param b: wing span
    :param S: wing area
    :param Lambda_half: Sweep angle at 0.5*chord
    :param eta: =0.95
    :return: Lift curve slope for tail AND wing using DATCOM method
    """
    if cruise==True:
        M = Speeds(conf)[-1]
    else:
        M= Speeds(conf)[2]
    beta = np.sqrt(1 - M ** 2)
    value = 2 * np.pi * A / (2 + np.sqrt(4 + ((A * beta / eta) ** 2) * (1 + (np.tan(Lambda_half) / beta) ** 2)))
    return value


def values_conf_1_2(conf,sens,sens_value):
    datafile = open(os.path.join(root_path, "data/inputs_config_%.0f.json" % (conf)), "r")
    data = json.load(datafile)
    datafile.close()
    lfus = data["Structures"]["l_fus"]
    hfus = data["Structures"]["h_fus"]
    wfus = data["Structures"]["w_fus"]
    CD0 = data["Aerodynamics"]["CDmin"]
    e = data["Aerodynamics"]["e"]
    if sens==False:
        Afwd = data["Aerodynamics"]["AR"] * 2
    else:
        Afwd = data["Aerodynamics"]["AR"] * 2*(1+sens_value/100)
    Sweep_c4_fwd = data["Aerodynamics"]["Sweep_front"]
    Sweep_c4_rear = data["Aerodynamics"]["Sweep_back"]
    b_fwd = np.sqrt(Afwd * data["Aerodynamics"]["S_front"])
    b_rear = np.sqrt(Afwd * data["Aerodynamics"]["S_back"])
    S_fwd = data["Aerodynamics"]["S_front"]
    S_rear = data["Aerodynamics"]["S_back"]

    CLa_fwd = data["Aerodynamics"]["CLalpha_back"]
    CLa_rear = CLa_fwd
    Cm_ac_fwd = data["Aerodynamics"]["Cm_ac_front"]
    Cm_ac_rear = data["Aerodynamics"]["Cm_ac_back"]
    CL_max_fwd = data["Aerodynamics"]["CLmax_front"]
    CL_max_rear = data["Aerodynamics"]["CLmax_back"]
    c_fwd = data["Aerodynamics"]["MAC1"]
    c_rear = data["Aerodynamics"]["MAC2"]
    cr =  data["Aerodynamics"]["c_r"]
    xacfwd = 0.25 * c_fwd
    xacrear = lfus - (1 - 0.25) * c_rear
    values = [CL_max_fwd, CL_max_rear, Cm_ac_fwd, Cm_ac_rear, CLa_fwd, CLa_rear, S_fwd, S_rear, Afwd, c_fwd, c_rear,
                      b_fwd, b_rear, xacfwd, xacrear, e, CD0, lfus, hfus, wfus,Sweep_c4_fwd,Sweep_c4_rear,cr]
    return values

def values_sens_1_2(conf,cruise,sens,sens_value,AR_t,AR,CL_t):
    datafile = open(os.path.join(root_path, "data/inputs_config_%.0f.json" % (conf)), "r")
    data = json.load(datafile)
    datafile.close()
    lfus = data["Structures"]["l_fus"]
    hfus = data["Structures"]["h_fus"]
    wfus = data["Structures"]["w_fus"]
    CD0 = data["Aerodynamics"]["CDmin"]
    e = data["Aerodynamics"]["e"]
    if sens==False:
        Afwd = data["Aerodynamics"]["AR"] * 2
    else:
        Afwd = data["Aerodynamics"]["AR"] * 2*(1+sens_value/100)
    if AR_t == True:
        Afwd = AR
    else:
        Afwd = Afwd
    Sweep_c4_fwd = data["Aerodynamics"]["Sweep_front"]
    Sweep_c4_rear = data["Aerodynamics"]["Sweep_back"]
    taper = 0.4
    S_fwd = data["Aerodynamics"]["S_front"]
    S_rear = data["Aerodynamics"]["S_back"]
    S = S_rear+S_fwd
    b_fwd = np.sqrt(Afwd * data["Aerodynamics"]["S_front"])
    b_rear = np.sqrt(Afwd * data["Aerodynamics"]["S_back"])
    c_r_fwd = 2 * S_fwd / ((1 + taper) * b_fwd)
    c_r_rear = 2 * S_rear / ((1 + taper) * b_rear)
    c_fwd = (2 / 3) * c_r_fwd * ((1 + taper + taper ** 2) / (1 + taper))
    c_rear = (2 / 3) * c_r_rear * ((1 + taper + taper ** 2) / (1 + taper))
    if cruise:
        CLa_fwd = C_L_a(conf,True,Afwd,Sweep_c4_fwd)
    else:
        CLa_fwd = C_L_a(conf, False, Afwd, Sweep_c4_fwd)
    CLa_rear = CLa_fwd # ASSUMES EQUAL TO AR
    Cm_ac_fwd = data["Aerodynamics"]["Cm_ac_front"]
    Cm_ac_rear = data["Aerodynamics"]["Cm_ac_back"]
    CL_max_fwd = data["Aerodynamics"]["CLmax_front"]

    if CL_t == True:
        CL_max_fwd *= (1+sense_value/100)
    else:
        CL_max_fwd = CL_max_fwd
    CL_max_rear = data["Aerodynamics"]["CLmax_back"]
    xacfwd = 0.25 * c_fwd
    xacrear = lfus - (1 - 0.25) * c_rear
    values = [CL_max_fwd, CL_max_rear, Cm_ac_fwd, Cm_ac_rear, CLa_fwd, CLa_rear, S_fwd, S_rear, Afwd, c_fwd, c_rear,
                      b_fwd, b_rear, xacfwd, xacrear, e, CD0, lfus, hfus, wfus,Sweep_c4_fwd,Sweep_c4_rear,c_r_fwd]
    return values

def cg_range_conf_1_2(conf,s,s_value,AR_t,AR,CL_t):
    conf = conf
    values_c = values_sens_1_2(conf,True,s,s_value,AR_t,AR,CL_t)
    CLfwd,CLrear,Cmacfwd,Cmacrear,CLafwd, CLarear,Sfwd,Srear,Afwd,cfwd,crear,b_fwd,b_rear,\
    xacfwd,xacrear,e, CD0,lfus,hfus,wfus,Sweep_c4_fwd,Sweep_c4_rear,cr = values_c
    # CLfwd = CLfwd*1.1
    #CDafwd = 2*CLafwd*CLfwd/(np.pi*Afwd*e)
    #CDarear = 2*CLarear*CLrear/(np.pi*Afwd*e)
    deda = deps_da(Sweep_c4_fwd, b_fwd,lh(xacfwd,xacrear), hfus, Afwd, CLafwd,conf)
    # print("de/da = ",deda)
    xacfwd_stab = 0.25*cfwd
    xacfwd_control = 0.25 * cfwd
    o = CLafwd*xacfwd_stab+CLarear*(lfus-0.75*cfwd)*Srear/Sfwd*(1-deda)
    p =CLafwd*1+CLarear*1*Srear/Sfwd*(1-deda)
    xcg_max = o/p
    #CLrear = 0.8*CLfwd
    oo = 1/cfwd*(CLfwd * xacfwd_control + CLrear * xacrear * Srear / Sfwd -Cmacrear*Srear/Sfwd*crear)-Cmacfwd
    pp = 1/cfwd*(CLfwd * 1 + CLrear * 1 * Srear / Sfwd)
    xcg_min = oo/pp
    if AR_t ==False and CL_t==False:
        print("Configuration %.0f range: %.4f < x_cg < %.4f"%(conf,xcg_min,xcg_max))
        print("CG Range =%.3f" % ((xcg_max - xcg_min)))
    return xcg_min, (xcg_max)

def values_conf_3(sens,sens_value, AR_t,AR):
    datafile = open(os.path.join(root_path, "data/inputs_config_3.json"), "r")
    data = json.load(datafile)
    datafile.close()
    CD0 = data["Aerodynamics"]["CDmin"]
    e = data["Aerodynamics"]["e"]
    CLa_fwd = data["Aerodynamics"]["CLalpha_back"]
    lfus = data["Structures"]["l_fus"]
    hfus = data["Structures"]["h_fus"]
    wfus = data["Structures"]["w_fus"]
    CD0 = data["Aerodynamics"]["CDmin"]
    e = data["Aerodynamics"]["e"]
    if sens == False:
        Afwd = data["Aerodynamics"]["AR"] * 2
    else:
        Afwd = data["Aerodynamics"]["AR"] * 2 * (1 + sens_value / 100)

    if AR_t == True:
        Afwd = AR
    else:
        Afwd = Afwd
    Sweep_c4_fwd = data["Aerodynamics"]["Sweep_front"]
    Sweep_c4_rear = data["Aerodynamics"]["Sweep_back"]
    taper = 0.4
    S_fwd = data["Aerodynamics"]["S_front"]
    S_rear = data["Aerodynamics"]["S_back"]
    S = S_rear + S_fwd
    b_fwd = np.sqrt(Afwd * data["Aerodynamics"]["S_front"])
    b_rear = np.sqrt(Afwd * data["Aerodynamics"]["S_back"])
    c_r_fwd = 2 * S_fwd / ((1 + taper) * b_fwd)
    c = (2 / 3) * c_r_fwd * ((1 + taper + taper ** 2) / (1 + taper))
    cruise=True
    if cruise:
        CLa_fwd = C_L_a(3, True, Afwd, Sweep_c4_fwd)
    else:
        CLa_fwd = C_L_a(3, False, Afwd, Sweep_c4_fwd)
    CLa_rear = CLa_fwd
    Cm_ac_fwd = data["Aerodynamics"]["Cm_ac_front"]
    Cm_ac_rear = data["Aerodynamics"]["Cm_ac_back"]
    CL_max_fwd = data["Aerodynamics"]["CLmax_front"]
    CL_max_rear = data["Aerodynamics"]["CLmax_back"]
    # c = data["Aerodynamics"]["MAC1"]
    values = [CL_max_fwd, CL_max_rear, Cm_ac_fwd, Cm_ac_rear, CLa_fwd, CLa_rear, S_fwd, S_rear, Afwd, c, b_fwd,e,CD0]
    return values

def cg_range_conf_3(sens,sens_value, AR_t,AR,eta):
    values = values_conf_3(sens,sens_value,AR_t,AR)
    CLfwd, CLrear, Cmacfwd, Cmacrear, CLafwd, CLarear, Sfwd, Srear, Afwd, cfwd, bfwd,e,CD0 = values
    lfus = 4
    hfus = 1.6
    if sens==False:
        Afwd = Afwd
    else:
        Afwd = Afwd*(1+sens_value/100)

    if AR_t == True:
        Afwd = AR
    else:
        Afwd = Afwd
    xacfwd_stab = lfus/2-cfwd/2 + 0.263*cfwd
    xacfwd_control = lfus/2-cfwd/2 +  0.263*cfwd
    CD = CD0 + CLfwd**2/(np.pi*Afwd*e)
    zaccg = eta/100*hfus
    xcg_max = xacfwd_stab
    xcg_min = xacfwd_control-Cmacfwd/CLfwd*cfwd-CD/CLfwd*zaccg
    if AR_t == False and eta[0] ==0:
        print("Configuration 3 range: %.4f < x_cg < %.4f"%(xcg_min[0],xcg_max))
        # print("CG Range =%.3f"%((xcg_max-xcg_min)))
    return xcg_min, (xcg_max)

AR = np.linspace(1,15,100)

eta = np.linspace(0,50,1000)
cg1 = cg_range_conf_1_2(1,s=False,s_value=0,AR_t =False,AR=AR,CL_t = False)
cg2 = cg_range_conf_1_2(2,s=False,s_value=0,AR_t =False,AR=AR,CL_t = False)
cg3 = cg_range_conf_3(sens=False,sens_value=0,AR_t =False,AR=AR,eta=np.array([0]))
cg3_1 = cg_range_conf_3(sens=False,sens_value=0,AR_t =False,AR=AR,eta=eta)[0]

print("-----------------------------------------------------------------")
print("---------------------------GRAPHS--------------------------------")
print("-----------------------------------------------------------------")
plt.plot(eta,cg3_1)
plt.xlabel(r"$\eta$ [%]",fontsize=14)
plt.ylabel(r"$x_{cg_{min}}$ [m]",fontsize=14)
plt.show()

sense_value = np.linspace(0.001,100,1e3)
# cg1_1 = cg_range_conf_1_2(1,s=True,s_value=sense_value,AR_t =False,AR=AR,CL_t = False)
# plt.plot(sense_value*100,cg1_1)
# plt.xlabel("Increase in AR [%]")
# plt.ylabel("CG range [m]")
# plt.show()

cg_CL = cg_range_conf_1_2(1,s=True,s_value=sense_value,AR_t =False,AR=AR,CL_t = True)[0]
plt.plot(sense_value,cg_CL)
plt.xlabel(r"Increase in $C_{L_{fwd}}$ [%]",fontsize=14)
plt.ylabel(r"$x_{cg_{min}}$ [m]",fontsize=14)
plt.show()

cg_CL = cg_range_conf_1_2(2,s=True,s_value=sense_value,AR_t =False,AR=AR,CL_t = True)[0]
plt.plot(sense_value,cg_CL)
plt.xlabel(r"Increase in $C_{L_{fwd}}$ [%]",fontsize=14)
plt.ylabel(r"$x_{cg_{min}}$ [m]",fontsize=14)
plt.show()

# cg1_1 = cg_range_conf_1_2(1,s=True,s_value=-sense_value)
# plt.plot(sense_value*100,cg1_1)
# plt.xlabel("Decrease in AR [%]")
# plt.ylabel("CG range [m]")
# plt.show()

CLa = C_L_a(1,True,values_sens_1_2(1,True,True,sense_value,AR_t =False,AR=AR,CL_t = False)[8],0)
plt.plot(sense_value,CLa)
plt.xlabel("Increase in AR [%]")
plt.ylabel(r"$C_{L_{\alpha}}$ [1/rad]")
plt.show()

AR = np.linspace(1,20,1000)
cgAR = cg_range_conf_1_2(1,s=True,s_value=0,AR_t =True,AR=AR,CL_t = False)[1]
plt.plot(AR,cgAR)
plt.xlabel("AR [-]",fontsize=14)
plt.ylabel(r"$x_{cg_{max}}$ [m]",fontsize=14)
plt.show()

cgAR_3 = cg_range_conf_3(sens=True,sens_value=0,AR_t =True,AR=AR,eta=0)[1]
plt.plot(AR,cgAR_3)
plt.xlabel("AR [-]",fontsize=14)
plt.ylabel(r"$x_{cg_{max}}$ [m]",fontsize=14)
plt.show()

# CLa = C_L_a(1,True,AR,0)
# plt.plot(AR,CLa)
# plt.xlabel(" AR [-]")
# plt.ylabel(r"$C_{L_{\alpha}}$ [1/rad]")
# plt.show()


# cg1_1 = cg_range_conf_1_2(1,s=True,s_value=12)

# cg2 = cg_range_conf_1_2(2,s=False,s_value=0)
# # cg2 = cg_range_conf_1_2(2,s=True,s_value=5)
# cg3 = cg_range_conf_3()
# cg3_1 = cg_range_conf_3(eta=0)

def sensitivity(in1,in2):
    return (in1-in2)/in1*100

# print("-----------------------------------------------------------------")
# print("--------------------SENSITIVITY ANALYSIS-------------------------")
# print("-----------------------------------------------------------------")

# print("Configuration 1: Initial sensitivity analysis on CG range for change in AR", sensitivity(cg1,cg1_1))
# print("Configuration 3: Initial sensitivity analysis on CG range for change in eta", sensitivity(cg1,cg3_1))


print("-----------------------------------------------------------------")
print("--------------------LATERAL STABILITY----------------------------")
print("-----------------------------------------------------------------")
def est_Cnbeta(conf):
    def values_conf_1_2(conf):
        datafile = open(os.path.join(root_path, "data/inputs_config_%.0f.json" % (conf)), "r")
        data = json.load(datafile)
        datafile.close()
        MTOW = data["Structures"]["MTOW"]
        lfus = data["Structures"]["l_fus"]
        hfus = data["Structures"]["h_fus"]
        wfus = data["Structures"]["w_fus"]
        CD0 = data["Aerodynamics"]["CDmin"]
        e = data["Aerodynamics"]["e"]
        if conf==3:
            Afwd = data["Aerodynamics"]["AR"]
            Arear = data["Aerodynamics"]["AR"]
        else:
            Afwd = data["Aerodynamics"]["AR"]*2
            Arear = data["Aerodynamics"]["AR"]*2
        Sweep_c4_fwd = data["Aerodynamics"]["Sweep_front"]
        Sweep_c4_rear = data["Aerodynamics"]["Sweep_back"]
        b_fwd = np.sqrt(data["Aerodynamics"]["AR"] * data["Aerodynamics"]["S_front"])
        b_rear = np.sqrt(data["Aerodynamics"]["AR"] * data["Aerodynamics"]["S_back"])
        S_fwd = data["Aerodynamics"]["S_front"]
        S_rear = data["Aerodynamics"]["S_back"]
        CLa_fwd = data["Aerodynamics"]["CLalpha_back"]
        CLa_rear = CLa_fwd
        Cm_ac_fwd = data["Aerodynamics"]["Cm_ac_front"]
        Cm_ac_rear = data["Aerodynamics"]["Cm_ac_back"]
        CL_max_fwd = data["Aerodynamics"]["CLmax_front"]
        CL_max_rear = data["Aerodynamics"]["CLmax_back"]
        c_fwd = data["Aerodynamics"]["MAC1"]
        c_rear = data["Aerodynamics"]["MAC2"]
        cr =  data["Aerodynamics"]["c_r"]
        xacfwd = 0.25 * c_fwd
        xacrear = lfus - (1 - 0.25) * c_rear
        if conf==3:
            Sref = S_fwd
        else:
            Sref = S_fwd+S_rear
        CLdes = MTOW/(0.5*1.2*Speeds(conf)[0]**2*Sref)
        values = [CL_max_fwd, CL_max_rear, CLdes, Cm_ac_fwd, Cm_ac_rear, CLa_fwd, CLa_rear, S_fwd, S_rear, Afwd, Arear,c_fwd, c_rear,
                  b_fwd, b_rear, xacfwd, xacrear, e, CD0, lfus, hfus, wfus,Sweep_c4_fwd,Sweep_c4_rear,cr,Sref]
        return values

    CL_max_fwd, CL_max_rear, CLdes, Cm_ac_fwd, Cm_ac_rear, CLa_fwd, CLa_rear, S_fwd, S_rear, Afwd, Arear, c_fwd, c_rear,\
    b_fwd, b_rear, xacfwd, xacrear, e, CD0, lfus, hfus, wfus, Sweep_c4_fwd, Sweep_c4_rear, cr,Sref = values_conf_1_2(conf)
    V_fus = 4*np.pi/3*wfus*lfus*hfus
    # V_fus = wfus*lfus*hfus
    Cnb_fus_fwd = -1.3 * (V_fus * lfus / wfus) * (1 / (S_fwd * b_fwd))
    if conf == 3:
        Cnb_w_rear = 0
        Cnb_fus = Cnb_fus_fwd
    else:
        Cnb_w_rear = CLdes ** 2 * (1 / (4 * np.pi * Arear))
        Cnb_fus_rear = -1.3 * (V_fus * lfus / wfus) * (1 / (S_rear * b_rear))
        Cnb_fus = (Cnb_fus_fwd + Cnb_fus_rear) / 2
    print("AR = %.1f "%(Afwd))
    Cnb_w_fwd = CLdes**2*(1/(4*np.pi*Afwd)) #assumes 0 sweep
    Cnb_w = Cnb_w_fwd+Cnb_w_rear
    if conf==1 or conf==2:
        lv = lfus-1.682
        Cnb_fus *= 1/100
        Cnb = Cnb_w + Cnb_fus
    else:
        lv = lfus-1.7947
        Cnb_fus *= 1/10
        Cnb = Cnb_w + Cnb_fus
    bv = np.sqrt(2 * abs(Cnb) * S_fwd * b_fwd / (np.pi * lv))
    print("Wing: Cnb_w = %.6f 1/rad"%(Cnb_w))
    print("Fuselage: Cnb_fus = %.6f 1/rad" % (Cnb_fus))
    print("Total: Cnb = %.6f 1/rad" % (Cnb))
    print("Required vertical stabiliser bv = %.4f [m]"%(bv))
    return Cnb

est_Cnbeta(1)
est_Cnbeta(2)
est_Cnbeta(3)

print("-----------------------------------------------------------------")
print("--------------------PULL-UP MANOEUVRE----------------------------")
print("-----------------------------------------------------------------")

def CmqCzq(conf,sens,sens_value,AR_t,AR):
    CL_t = False
    if conf == 1 or conf ==2:
        xcg = 1.6853
        cruise = True
        values = values_sens_1_2(conf,cruise,sens,sens_value,AR_t,AR,CL_t)
        CLfwd, CLrear, Cmacfwd, Cmacrear, CLafwd, CLarear, Sfwd, Srear, Afwd, cfwd, crear, b_fwd, b_rear, \
        xacfwd, xacrear, e, CD0, lfus, hfus, wfus, Sweep_c4_fwd, Sweep_c4_rear, cr = values
        S = Sfwd+Srear
        print("x_ac_fwd = %.3f [m] and x_ac_rear = %.3f [m]"%(xacfwd,xacrear))
        Czq = (CLafwd*(xcg-xacfwd)/cfwd)-CLarear*Srear/Sfwd*(xacrear-xcg)/cfwd
        Cmq = -(CLafwd*Sfwd/S*(xcg-xacfwd)**2/cfwd**2+CLarear*Srear/S*(xacrear-xcg)**2/cfwd**2)
    else:
        xcg = 1.7947
        values = values_conf_3(sens,sens_value,AR_t,AR)
        CL_max_fwd, CL_max_rear, Cm_ac_fwd, Cm_ac_rear, CLa_fwd, CLa_rear, S_fwd, S_rear, Afwd, c, b_fwd, e, CD0 = values
        lfus = 4
        hfus = 1.6
        if sens == False:
            Afwd = Afwd
        else:
            Afwd = Afwd * (1 + sens_value / 100)

        if AR_t == True:
            Afwd = AR
        else:
            Afwd = Afwd
        xacfwd = lfus / 2 - c/ 2 + 0.26 * c
        print("x_ac = %.3f [m]" % (xacfwd))
        Czq = -CLa_fwd*(xacfwd-xcg)/c
        Cmq = -Czq*(xacfwd-xcg)/c
    print("Configuration %.0f: C_Z_q = %.3f 1/rad and C_m_q = %.3f 1/rad"%(conf,Czq,Cmq))
    return Czq,Cmq

CmqCzq(1,sens=True,sens_value=0,AR_t=False,AR=AR)
CmqCzq(2,sens=True,sens_value=0,AR_t=False,AR=AR)
CmqCzq(3,sens=True,sens_value=0,AR_t=False,AR=AR)

def Sr_Sfwd(conf,s,s_value,AR_t,AR,CL_t,Xcg,d):
    conf = conf
    values_c = values_sens_1_2(conf, True, s, s_value, AR_t, AR, CL_t)
    CLfwd, CLrear, Cmacfwd, Cmacrear, CLafwd, CLarear, Sfwd, Srear, Afwd, cfwd, crear, b_fwd, b_rear, \
    xacfwd, xacrear, e, CD0, lfus, hfus, wfus, Sweep_c4_fwd, Sweep_c4_rear, cr = values_c
    CLfwd = CLfwd*1.4
    CDfwd = CD0+ CLfwd**2/(np.pi*Afwd*e)
    CDrear = CD0 + CLrear ** 2 / (np.pi * Afwd * e)
    c = Sfwd/(Sfwd+Srear)*cfwd+ Srear/(Srear+Sfwd)*crear
    # CDafwd = 2*CLafwd*CLfwd/(np.pi*Afwd*e)
    # CDarear = 2*CLarear*CLrear/(np.pi*Afwd*e)
    deda = deps_da(Sweep_c4_fwd, b_fwd, lh(xacfwd, xacrear), hfus, Afwd, CLafwd, conf)
    # print("de/da = ",deda)
    SrSfwd_stab = CLafwd*(Xcg-xacfwd)/(CLarear*(1-deda)*(xacrear-d-Xcg))
    SrSfwd_control = (-Cmacfwd*cfwd+CDfwd*hfus/2 -CLfwd*(Xcg-xacfwd-d)/(CDrear*hfus/2-CLrear*(xacrear-Xcg)+Cmacrear*crear))
    return SrSfwd_stab**-1,SrSfwd_control**-1
Xcg = np.linspace(0.75, 4,1000)
d = [0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1,1.25,1.5,1.75,2]
for i in range(0,len(d)):
    SrSfwd = Sr_Sfwd(1,s=False,s_value=0,AR_t =False,AR=AR,CL_t = False,Xcg=Xcg,d=d[i])
    plt.plot(Xcg,SrSfwd[0],label="Neutral stability")
    plt.plot(Xcg,SrSfwd[1],label="Controllability limit")
    plt.xlabel(r"${x}_{cg}$ [m]",fontsize=14)
    plt.ylabel(r"$S_{fwd}/S_{rear}$ [-]",fontsize=14)
    plt.ylim(0,max(SrSfwd[1]))
    plt.legend()
    plt.show()
